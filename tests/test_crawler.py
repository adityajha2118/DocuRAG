"""Tests for src/ingestion/crawler.py — all HTTP calls are mocked."""

from __future__ import annotations

import time
from unittest.mock import MagicMock, patch, PropertyMock

import pytest
import requests

from src.ingestion.crawler import (
    CrawlResult,
    normalize_url,
    is_url_allowed,
    extract_links,
    fetch_page,
    crawl_source,
    check_robots_txt,
)
from src.ingestion.sources import DocSource


# ── URL normalization ────────────────────────────────────────────────────

class TestNormalizeUrl:
    def test_strips_fragment(self):
        assert normalize_url("https://example.com/page#section") == \
               "https://example.com/page"

    def test_strips_query_string(self):
        assert normalize_url("https://example.com/page?q=hello") == \
               "https://example.com/page"

    def test_strips_trailing_slash(self):
        assert normalize_url("https://example.com/docs/") == \
               "https://example.com/docs"

    def test_preserves_root_slash(self):
        assert normalize_url("https://example.com/") == \
               "https://example.com/"

    def test_lowercases_scheme_and_host(self):
        assert normalize_url("HTTPS://Example.COM/Path") == \
               "https://example.com/Path"

    def test_identical_urls_normalize_same(self):
        a = normalize_url("https://example.com/docs/page#intro")
        b = normalize_url("https://example.com/docs/page#setup")
        assert a == b


# ── URL allowed check ───────────────────────────────────────────────────

SAMPLE_SOURCE = DocSource(
    name="Test",
    base_url="https://docs.example.com/stable/",
    allowed_domains=frozenset({"docs.example.com"}),
    allowed_prefixes=("/stable/",),
)


class TestIsUrlAllowed:
    def test_allowed_url(self):
        assert is_url_allowed("https://docs.example.com/stable/intro", SAMPLE_SOURCE)

    def test_wrong_domain(self):
        assert not is_url_allowed("https://other.com/stable/intro", SAMPLE_SOURCE)

    def test_wrong_prefix(self):
        assert not is_url_allowed("https://docs.example.com/v2/intro", SAMPLE_SOURCE)

    def test_root_not_allowed_if_prefix_is_subpath(self):
        assert not is_url_allowed("https://docs.example.com/", SAMPLE_SOURCE)


# ── Link extraction ─────────────────────────────────────────────────────

class TestExtractLinks:
    def test_extracts_absolute_links(self):
        html = '<html><body><a href="https://example.com/page">Link</a></body></html>'
        links = extract_links(html, "https://example.com/")
        assert "https://example.com/page" in links

    def test_resolves_relative_links(self):
        html = '<html><body><a href="subpage">Link</a></body></html>'
        links = extract_links(html, "https://example.com/docs/")
        assert "https://example.com/docs/subpage" in links

    def test_ignores_mailto_links(self):
        html = '<html><body><a href="mailto:a@b.com">Email</a></body></html>'
        links = extract_links(html, "https://example.com/")
        assert len(links) == 0

    def test_ignores_javascript_links(self):
        html = '<html><body><a href="javascript:void(0)">Click</a></body></html>'
        links = extract_links(html, "https://example.com/")
        assert len(links) == 0

    def test_multiple_links(self):
        html = """<html><body>
            <a href="/a">A</a>
            <a href="/b">B</a>
            <a href="/c">C</a>
        </body></html>"""
        links = extract_links(html, "https://example.com/")
        assert len(links) == 3


# ── fetch_page ───────────────────────────────────────────────────────────

class TestFetchPage:
    @patch("src.ingestion.crawler.requests.get")
    def test_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_get.return_value = mock_resp

        resp = fetch_page("https://example.com", max_retries=2)
        assert resp is not None
        assert resp.status_code == 200
        mock_get.assert_called_once()

    @patch("src.ingestion.crawler.requests.get")
    def test_retries_on_500(self, mock_get):
        err_resp = MagicMock()
        err_resp.status_code = 500

        ok_resp = MagicMock()
        ok_resp.status_code = 200

        mock_get.side_effect = [err_resp, ok_resp]

        resp = fetch_page("https://example.com", max_retries=2)
        assert resp is not None
        assert resp.status_code == 200
        assert mock_get.call_count == 2

    @patch("src.ingestion.crawler.requests.get")
    def test_retries_on_429(self, mock_get):
        err_resp = MagicMock()
        err_resp.status_code = 429

        ok_resp = MagicMock()
        ok_resp.status_code = 200

        mock_get.side_effect = [err_resp, ok_resp]

        resp = fetch_page("https://example.com", max_retries=2)
        assert resp.status_code == 200

    @patch("src.ingestion.crawler.requests.get")
    def test_gives_up_after_max_retries(self, mock_get):
        err_resp = MagicMock()
        err_resp.status_code = 500
        mock_get.return_value = err_resp

        resp = fetch_page("https://example.com", max_retries=2)
        assert resp is None
        assert mock_get.call_count == 3  # 1 initial + 2 retries

    @patch("src.ingestion.crawler.requests.get")
    def test_no_retry_on_404(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        resp = fetch_page("https://example.com", max_retries=2)
        assert resp is not None
        assert resp.status_code == 404
        mock_get.assert_called_once()

    @patch("src.ingestion.crawler.requests.get")
    def test_retries_on_timeout(self, mock_get):
        ok_resp = MagicMock()
        ok_resp.status_code = 200
        mock_get.side_effect = [requests.Timeout("timeout"), ok_resp]

        resp = fetch_page("https://example.com", max_retries=2)
        assert resp is not None
        assert resp.status_code == 200

    @patch("src.ingestion.crawler.requests.get")
    def test_returns_none_after_all_timeouts(self, mock_get):
        mock_get.side_effect = requests.Timeout("timeout")

        resp = fetch_page("https://example.com", max_retries=1)
        assert resp is None
        assert mock_get.call_count == 2


# ── robots.txt ───────────────────────────────────────────────────────────

class TestRobotsTxt:
    @patch("src.ingestion.crawler.requests.get")
    def test_allows_when_no_disallow(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "User-agent: *\nAllow: /"
        mock_get.return_value = mock_resp

        rp = check_robots_txt("https://example.com/docs/")
        assert rp.can_fetch("DocuRAG", "https://example.com/docs/page")

    @patch("src.ingestion.crawler.requests.get")
    def test_disallows_when_blocked(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.text = "User-agent: *\nDisallow: /docs/"
        mock_get.return_value = mock_resp

        rp = check_robots_txt("https://example.com/docs/")
        assert not rp.can_fetch("DocuRAG", "https://example.com/docs/page")

    @patch("src.ingestion.crawler.requests.get")
    def test_allows_when_robots_txt_missing(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.status_code = 404
        mock_get.return_value = mock_resp

        rp = check_robots_txt("https://example.com/")
        # 404 → allow all
        assert rp.can_fetch("DocuRAG", "https://example.com/any/path")

    @patch("src.ingestion.crawler.requests.get")
    def test_allows_on_network_error(self, mock_get):
        mock_get.side_effect = requests.ConnectionError("fail")

        rp = check_robots_txt("https://example.com/")
        assert rp.can_fetch("DocuRAG", "https://example.com/any")


# ── crawl_source ─────────────────────────────────────────────────────────

class TestCrawlSource:
    FASTAPI_SOURCE = DocSource(
        name="FastAPI",
        base_url="https://fastapi.tiangolo.com/",
        allowed_domains=frozenset({"fastapi.tiangolo.com"}),
        allowed_prefixes=("/",),
    )

    def _make_response(self, url: str, html: str, status: int = 200) -> MagicMock:
        resp = MagicMock()
        resp.status_code = status
        resp.url = url
        resp.text = html
        resp.headers = {"Content-Type": "text/html; charset=utf-8"}
        return resp

    @patch("src.ingestion.crawler.check_robots_txt")
    @patch("src.ingestion.crawler.fetch_page")
    def test_respects_max_pages(self, mock_fetch, mock_robots):
        rp = MagicMock()
        rp.can_fetch.return_value = True
        mock_robots.return_value = rp

        page_html = '<html><body><p>Page</p><a href="/other">Link</a></body></html>'
        mock_fetch.return_value = self._make_response(
            "https://fastapi.tiangolo.com/", page_html,
        )

        results = crawl_source(self.FASTAPI_SOURCE, max_pages=3, delay=0)
        assert len(results) <= 3

    @patch("src.ingestion.crawler.check_robots_txt")
    @patch("src.ingestion.crawler.fetch_page")
    def test_returns_empty_when_robots_disallows(self, mock_fetch, mock_robots):
        rp = MagicMock()
        rp.can_fetch.return_value = False
        mock_robots.return_value = rp

        results = crawl_source(self.FASTAPI_SOURCE, max_pages=5, delay=0)
        assert results == []
        mock_fetch.assert_not_called()

    @patch("src.ingestion.crawler.check_robots_txt")
    @patch("src.ingestion.crawler.fetch_page")
    def test_skips_non_html(self, mock_fetch, mock_robots):
        rp = MagicMock()
        rp.can_fetch.return_value = True
        mock_robots.return_value = rp

        resp = MagicMock()
        resp.status_code = 200
        resp.url = "https://fastapi.tiangolo.com/"
        resp.text = "not html"
        resp.headers = {"Content-Type": "application/pdf"}
        mock_fetch.return_value = resp

        results = crawl_source(self.FASTAPI_SOURCE, max_pages=5, delay=0)
        assert len(results) == 0

    @patch("src.ingestion.crawler.check_robots_txt")
    @patch("src.ingestion.crawler.fetch_page")
    def test_follows_in_scope_links(self, mock_fetch, mock_robots):
        rp = MagicMock()
        rp.can_fetch.return_value = True
        mock_robots.return_value = rp

        page1_html = '<html><body><a href="/tutorial/">Tutorial</a></body></html>'
        page2_html = '<html><body><p>Tutorial content</p></body></html>'

        resp1 = self._make_response("https://fastapi.tiangolo.com/", page1_html)
        resp2 = self._make_response("https://fastapi.tiangolo.com/tutorial/", page2_html)

        mock_fetch.side_effect = [resp1, resp2]

        results = crawl_source(self.FASTAPI_SOURCE, max_pages=5, delay=0)
        assert len(results) == 2
        urls = [r.url for r in results]
        assert "https://fastapi.tiangolo.com/" in urls
        assert "https://fastapi.tiangolo.com/tutorial/" in urls

    @patch("src.ingestion.crawler.check_robots_txt")
    @patch("src.ingestion.crawler.fetch_page")
    def test_does_not_follow_out_of_scope_links(self, mock_fetch, mock_robots):
        rp = MagicMock()
        rp.can_fetch.return_value = True
        mock_robots.return_value = rp

        html = '<html><body><a href="https://other.com/page">External</a></body></html>'
        resp = self._make_response("https://fastapi.tiangolo.com/", html)
        mock_fetch.return_value = resp

        results = crawl_source(self.FASTAPI_SOURCE, max_pages=5, delay=0)
        assert len(results) == 1  # only the start page

    @patch("src.ingestion.crawler.check_robots_txt")
    @patch("src.ingestion.crawler.fetch_page")
    def test_no_duplicate_visits(self, mock_fetch, mock_robots):
        rp = MagicMock()
        rp.can_fetch.return_value = True
        mock_robots.return_value = rp

        html = '''<html><body>
            <a href="/page">A</a>
            <a href="/page#section1">B</a>
            <a href="/page?q=1">C</a>
        </body></html>'''
        resp = self._make_response("https://fastapi.tiangolo.com/", html)
        page_resp = self._make_response(
            "https://fastapi.tiangolo.com/page",
            "<html><body><p>Page</p></body></html>",
        )
        mock_fetch.side_effect = [resp, page_resp]

        results = crawl_source(self.FASTAPI_SOURCE, max_pages=10, delay=0)
        # Start page + /page (only once despite 3 links normalising to same URL)
        assert len(results) == 2

    @patch("src.ingestion.crawler.check_robots_txt")
    @patch("src.ingestion.crawler.fetch_page")
    def test_skips_failed_fetches(self, mock_fetch, mock_robots):
        rp = MagicMock()
        rp.can_fetch.return_value = True
        mock_robots.return_value = rp

        mock_fetch.return_value = None  # simulates total failure

        results = crawl_source(self.FASTAPI_SOURCE, max_pages=5, delay=0)
        assert results == []
