"""
DocuRAG — web crawler.

BFS crawler that respects robots.txt, domain/path-prefix restrictions,
rate-limiting, page caps, timeouts, and exponential-backoff retries.

Public entry point: :func:`crawl_source`.
"""

from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass
from urllib.parse import urlparse, urljoin, urldefrag
from urllib.robotparser import RobotFileParser

import requests
from bs4 import BeautifulSoup

from src.config import (
    CRAWL_DELAY_SECONDS,
    CRAWL_MAX_RETRIES,
    CRAWL_TIMEOUT_SECONDS,
    MAX_PAGES_PER_SOURCE,
)
from src.ingestion.sources import DocSource

logger = logging.getLogger(__name__)

USER_AGENT = "DocuRAG/1.0 (+https://github.com/docurag; educational project)"


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class CrawlResult:
    """Successful fetch of a single page."""
    url: str                    # final URL after any redirects
    html: str
    status_code: int
    source_name: str


# ── URL helpers ──────────────────────────────────────────────────────────

def normalize_url(url: str) -> str:
    """Canonical form of *url* for deduplication.

    * Strips fragment (``#section``)
    * Strips query string (doc sites don't use meaningful query params)
    * Lowercases scheme + netloc
    * Strips trailing ``/`` (except bare domain root)
    """
    url, _ = urldefrag(url)
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    path = parsed.path
    # Strip trailing slash unless path is just "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")
    return f"{scheme}://{netloc}{path}"


def is_url_allowed(url: str, source: DocSource) -> bool:
    """Return True if *url* falls within *source*'s domain + prefix rules."""
    parsed = urlparse(url)
    if parsed.netloc not in source.allowed_domains:
        return False
    return any(parsed.path.startswith(pfx) for pfx in source.allowed_prefixes)


# ── robots.txt ───────────────────────────────────────────────────────────

def check_robots_txt(
    base_url: str,
    user_agent: str = USER_AGENT,
    timeout: int = CRAWL_TIMEOUT_SECONDS,
) -> RobotFileParser:
    """Fetch and parse ``robots.txt`` for the domain of *base_url*.

    If robots.txt cannot be retrieved the parser defaults to *allow all*
    (standard behaviour per RFC 9309 §2.3).
    """
    parsed = urlparse(base_url)
    robots_url = f"{parsed.scheme}://{parsed.netloc}/robots.txt"
    rp = RobotFileParser()
    rp.set_url(robots_url)
    try:
        resp = requests.get(robots_url, timeout=timeout,
                            headers={"User-Agent": user_agent})
        if resp.status_code == 200:
            rp.parse(resp.text.splitlines())
        else:
            # Non-200 → treat as "allow all"
            rp.parse([])
    except (requests.RequestException, Exception) as exc:
        logger.warning("Could not fetch %s: %s — assuming allow-all", robots_url, exc)
        rp.parse([])
    return rp


# ── Single-page fetch with retry ────────────────────────────────────────

def fetch_page(
    url: str,
    timeout: int = CRAWL_TIMEOUT_SECONDS,
    max_retries: int = CRAWL_MAX_RETRIES,
) -> requests.Response | None:
    """GET *url* with exponential-backoff retry on transient failures.

    Returns the :class:`~requests.Response` on success (any status), or
    ``None`` if all attempts are exhausted.  Only **transient** failures
    (5xx, 429, connection/timeout errors) trigger retries.
    """
    headers = {"User-Agent": USER_AGENT}
    for attempt in range(max_retries + 1):
        try:
            resp = requests.get(url, timeout=timeout, headers=headers,
                                allow_redirects=True)

            # 4xx (except 429) — not transient, don't retry
            if 400 <= resp.status_code < 500 and resp.status_code != 429:
                return resp

            # Transient server / rate-limit errors
            if resp.status_code == 429 or resp.status_code >= 500:
                if attempt < max_retries:
                    wait = 2 ** attempt
                    logger.warning(
                        "Transient HTTP %d on %s — retrying in %ds (attempt %d/%d)",
                        resp.status_code, url, wait, attempt + 1, max_retries + 1,
                    )
                    time.sleep(wait)
                    continue
                logger.error(
                    "Giving up on %s after %d attempts (last status %d)",
                    url, max_retries + 1, resp.status_code,
                )
                return None

            # 2xx / 3xx — success
            return resp

        except (requests.Timeout, requests.ConnectionError) as exc:
            if attempt < max_retries:
                wait = 2 ** attempt
                logger.warning(
                    "Request error on %s: %s — retrying in %ds (attempt %d/%d)",
                    url, exc, wait, attempt + 1, max_retries + 1,
                )
                time.sleep(wait)
            else:
                logger.error("Giving up on %s after %d attempts: %s",
                             url, max_retries + 1, exc)
                return None

    return None  # pragma: no cover — loop always returns


# ── Link extraction ─────────────────────────────────────────────────────

def extract_links(html: str, base_url: str) -> list[str]:
    """Return all absolute ``href`` links found in *html*."""
    soup = BeautifulSoup(html, "lxml")
    links: list[str] = []
    for a in soup.find_all("a", href=True):
        href: str = a["href"]
        # Skip non-http(s) schemes
        if href.startswith(("mailto:", "javascript:", "tel:", "ftp:")):
            continue
        abs_url = urljoin(base_url, href)
        links.append(abs_url)
    return links


# ── BFS crawl ────────────────────────────────────────────────────────────

def crawl_source(
    source: DocSource,
    max_pages: int | None = None,
    delay: float | None = None,
) -> list[CrawlResult]:
    """BFS-crawl *source* up to *max_pages* pages.

    Parameters
    ----------
    source : DocSource
        Which documentation site to crawl.
    max_pages : int, optional
        Override for ``MAX_PAGES_PER_SOURCE``.
    delay : float, optional
        Override for ``CRAWL_DELAY_SECONDS``.

    Returns
    -------
    list[CrawlResult]
        Successfully fetched pages.  Empty list if robots.txt disallows.
    """
    if max_pages is None:
        max_pages = MAX_PAGES_PER_SOURCE
    if delay is None:
        delay = CRAWL_DELAY_SECONDS

    logger.info("Crawl starting — source=%s  base_url=%s  max_pages=%d",
                source.name, source.base_url, max_pages)

    # ── 1. Respect robots.txt ──
    rp = check_robots_txt(source.base_url)
    if not rp.can_fetch(USER_AGENT, source.base_url):
        logger.warning(
            "SKIPPING %s — robots.txt disallows crawling %s",
            source.name, source.base_url,
        )
        return []

    # ── 2. BFS state ──
    results: list[CrawlResult] = []
    visited: set[str] = set()             # normalized URLs already queued
    queue: deque[str] = deque()

    start_norm = normalize_url(source.base_url)
    visited.add(start_norm)
    queue.append(source.base_url)

    # Track whether the base_url itself matches allowed prefixes.
    # If not, we still fetch it for link discovery but don't save it.
    base_url_matches_prefixes = is_url_allowed(source.base_url, source)

    last_request_time: float = 0.0

    # ── 3. BFS loop ──
    while queue and len(results) < max_pages:
        url = queue.popleft()

        # robots.txt per-URL check
        if not rp.can_fetch(USER_AGENT, url):
            logger.debug("Skipping %s — disallowed by robots.txt", url)
            continue

        # Rate limiting
        elapsed = time.time() - last_request_time
        if elapsed < delay:
            time.sleep(delay - elapsed)

        # Fetch
        resp = fetch_page(url)
        last_request_time = time.time()

        if resp is None:
            continue

        # Handle redirects: check that the *final* URL is still in-scope
        final_url = resp.url
        if final_url != url:
            final_norm = normalize_url(final_url)
            if not is_url_allowed(final_url, source):
                # Still extract links if this was the base_url (for discovery)
                if url == source.base_url:
                    for link in extract_links(resp.text, final_url):
                        link_norm = normalize_url(link)
                        if link_norm not in visited and is_url_allowed(link, source):
                            visited.add(link_norm)
                            queue.append(link)
                logger.debug("Redirect %s → %s is out of scope, skipping",
                             url, final_url)
                continue
            if final_norm in visited:
                logger.debug("Redirect %s → %s already visited, skipping",
                             url, final_url)
                continue
            visited.add(final_norm)

        if resp.status_code != 200:
            logger.debug("Non-200 (%d) for %s, skipping", resp.status_code, url)
            continue

        # Content-type guard
        ctype = resp.headers.get("Content-Type", "")
        if "text/html" not in ctype.lower():
            logger.debug("Skipping non-HTML content at %s (%s)", url, ctype)
            continue

        # Only save pages that match allowed prefixes (or base_url if it matches)
        is_base = (normalize_url(url) == start_norm)
        if is_base and not base_url_matches_prefixes:
            # Base URL doesn't match prefixes — use it only for link discovery
            logger.info("Base URL %s used for link discovery only (not in allowed prefixes)",
                        final_url)
        else:
            results.append(CrawlResult(
                url=final_url,
                html=resp.text,
                status_code=resp.status_code,
                source_name=source.name,
            ))
            logger.info("[%d/%d] Fetched %s", len(results), max_pages, final_url)

        # Discover links (always, even from base URL used for discovery only)
        for link in extract_links(resp.text, final_url):
            link_norm = normalize_url(link)
            if link_norm not in visited and is_url_allowed(link, source):
                visited.add(link_norm)
                queue.append(link)

    logger.info("Crawl complete — source=%s  pages=%d", source.name, len(results))
    return results

