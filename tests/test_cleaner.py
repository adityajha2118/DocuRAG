"""Tests for src/ingestion/cleaner.py — no network access."""

import pytest
from bs4 import BeautifulSoup

from src.ingestion.cleaner import clean_html


# ── Removal tests ────────────────────────────────────────────────────────

def test_removes_script_tags():
    html = "<html><body><p>Hello</p><script>alert(1)</script></body></html>"
    soup = clean_html(html)
    assert soup.find("script") is None
    assert "Hello" in soup.get_text()


def test_removes_style_tags():
    html = "<html><body><style>.x{color:red}</style><p>Keep</p></body></html>"
    soup = clean_html(html)
    assert soup.find("style") is None
    assert "Keep" in soup.get_text()


def test_removes_nav_tags():
    html = "<html><body><nav><a href='/'>Home</a></nav><p>Content</p></body></html>"
    soup = clean_html(html)
    assert soup.find("nav") is None
    assert "Content" in soup.get_text()


def test_removes_footer_tags():
    html = "<html><body><p>Body</p><footer>© 2024</footer></body></html>"
    soup = clean_html(html)
    assert soup.find("footer") is None
    assert "Body" in soup.get_text()


def test_removes_elements_by_class_sidebar():
    html = '<html><body><div class="sidebar">Nav</div><p>Main</p></body></html>'
    soup = clean_html(html)
    assert "Nav" not in soup.get_text()
    assert "Main" in soup.get_text()


def test_removes_elements_by_class_navbar():
    html = '<html><body><div class="top-navbar">Links</div><p>Article</p></body></html>'
    soup = clean_html(html)
    assert "Links" not in soup.get_text()


def test_removes_elements_by_id():
    html = '<html><body><div id="cookie-consent">Accept</div><p>Keep</p></body></html>'
    soup = clean_html(html)
    assert "Accept" not in soup.get_text()
    assert "Keep" in soup.get_text()


def test_removes_elements_by_aria_role():
    html = '<html><body><div role="navigation">Menu</div><p>Text</p></body></html>'
    soup = clean_html(html)
    assert "Menu" not in soup.get_text()
    assert "Text" in soup.get_text()


def test_removes_cookie_banner():
    html = '<html><body><div class="cookie-banner">Cookies</div><p>OK</p></body></html>'
    soup = clean_html(html)
    assert "Cookies" not in soup.get_text()


def test_removes_search_modal():
    html = '<html><body><div class="search-modal">Search</div><p>Content</p></body></html>'
    soup = clean_html(html)
    assert "Search" not in soup.get_text()


def test_removes_mkdocs_material_chrome():
    html = """
    <html><body>
        <header class="md-header">Header</header>
        <div class="md-sidebar">Side</div>
        <div class="md-content"><p>Real content</p></div>
        <footer class="md-footer">Footer</footer>
    </body></html>
    """
    soup = clean_html(html)
    text = soup.get_text()
    assert "Real content" in text
    assert "Header" not in text
    assert "Side" not in text
    assert "Footer" not in text


def test_removes_html_comments():
    html = "<html><body><!-- TODO: fix --><p>Visible</p></body></html>"
    soup = clean_html(html)
    assert "TODO" not in str(soup)
    assert "Visible" in soup.get_text()


def test_removes_noscript():
    html = "<html><body><noscript>No JS</noscript><p>Hi</p></body></html>"
    soup = clean_html(html)
    assert soup.find("noscript") is None


def test_removes_svg():
    html = '<html><body><svg><path d="M0 0"/></svg><p>Text</p></body></html>'
    soup = clean_html(html)
    assert soup.find("svg") is None


# ── Preservation tests ───────────────────────────────────────────────────

def test_preserves_headings():
    html = "<html><body><h1>Title</h1><h2>Section</h2><p>Text</p></body></html>"
    soup = clean_html(html)
    assert soup.find("h1") is not None
    assert "Title" in soup.get_text()
    assert "Section" in soup.get_text()


def test_preserves_paragraphs():
    html = "<html><body><p>Paragraph one.</p><p>Paragraph two.</p></body></html>"
    soup = clean_html(html)
    assert soup.find_all("p") is not None
    assert "Paragraph one" in soup.get_text()


def test_preserves_code_blocks():
    html = '<html><body><pre><code class="language-python">x = 1</code></pre></body></html>'
    soup = clean_html(html)
    assert soup.find("pre") is not None
    assert "x = 1" in soup.get_text()


def test_preserves_lists():
    html = "<html><body><ul><li>A</li><li>B</li></ul></body></html>"
    soup = clean_html(html)
    assert soup.find("ul") is not None
    assert "A" in soup.get_text()


def test_preserves_tables():
    html = "<html><body><table><tr><td>Data</td></tr></table></body></html>"
    soup = clean_html(html)
    assert soup.find("table") is not None
    assert "Data" in soup.get_text()


def test_removes_nested_nav_inside_removed_parent():
    """Nested chrome inside a removed parent should also be removed."""
    html = """
    <html><body>
        <div class="sidebar"><nav><a href="/">Link</a></nav></div>
        <p>Keep me</p>
    </body></html>
    """
    soup = clean_html(html)
    assert "Link" not in soup.get_text()
    assert "Keep me" in soup.get_text()


def test_header_as_body_child_removed():
    """A <header> that is a direct child of <body> is treated as site header."""
    html = "<html><body><header><h1>Site Title</h1></header><p>Content</p></body></html>"
    soup = clean_html(html)
    assert "Site Title" not in soup.get_text()
    assert "Content" in soup.get_text()


def test_header_inside_article_preserved():
    """A <header> nested inside an article/section is NOT removed
    unless it matches class/id patterns."""
    html = """
    <html><body><article>
        <header><h2>Article Heading</h2></header>
        <p>Article body</p>
    </article></body></html>
    """
    soup = clean_html(html)
    text = soup.get_text()
    assert "Article Heading" in text
    assert "Article body" in text
