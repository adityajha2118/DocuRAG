"""Tests for src/ingestion/parser.py — no network access."""

import pytest

from src.ingestion.parser import (
    ParsedPage,
    ParsedSection,
    parse_page,
    _extract_title,
    _extract_canonical_url,
    _find_main_content,
    _container_to_text,
    _split_into_sections,
    _list_to_text,
    _table_to_text,
)
from bs4 import BeautifulSoup


# ── Sample HTML fixtures ─────────────────────────────────────────────────

SIMPLE_HTML = """
<html>
<head>
    <title>Test Page — Docs</title>
    <link rel="canonical" href="https://example.com/docs/test" />
</head>
<body>
<nav>Skip this</nav>
<article>
    <h1>Getting Started</h1>
    <p>Welcome to the docs.</p>
    <h2>Installation</h2>
    <p>Run <code>pip install example</code> to install.</p>
    <pre><code class="language-python">import example
example.run()
</code></pre>
    <h2>Usage</h2>
    <p>Here is how to use it.</p>
    <ul>
        <li>Step one</li>
        <li>Step two</li>
    </ul>
</article>
</body>
</html>
"""

TABLE_HTML = """
<html><body><article>
<h1>API Reference</h1>
<table>
    <tr><th>Param</th><th>Type</th><th>Description</th></tr>
    <tr><td>name</td><td>str</td><td>The name</td></tr>
    <tr><td>age</td><td>int</td><td>The age</td></tr>
</table>
</article></body></html>
"""

DL_HTML = """
<html><body><article>
<h1>Glossary</h1>
<dl>
    <dt>API</dt>
    <dd>Application Programming Interface</dd>
    <dt>REST</dt>
    <dd>Representational State Transfer</dd>
</dl>
</article></body></html>
"""

BLOCKQUOTE_HTML = """
<html><body><article>
<h1>Notes</h1>
<blockquote>This is important.</blockquote>
</article></body></html>
"""


# ── Title extraction ─────────────────────────────────────────────────────

def test_extract_title_from_h1():
    page = parse_page("https://example.com", SIMPLE_HTML, "Test")
    assert page.title == "Getting Started"


def test_extract_title_fallback_to_title_tag():
    html = "<html><head><title>Fallback Title</title></head><body><p>No h1</p></body></html>"
    page = parse_page("https://example.com", html, "Test")
    assert page.title == "Fallback Title"


def test_extract_title_untitled():
    html = "<html><body><p>Nothing</p></body></html>"
    page = parse_page("https://example.com", html, "Test")
    assert page.title == "Untitled"


# ── Canonical URL ────────────────────────────────────────────────────────

def test_canonical_url_from_link_tag():
    page = parse_page("https://example.com", SIMPLE_HTML, "Test")
    assert page.canonical_url == "https://example.com/docs/test"


def test_canonical_url_missing():
    html = "<html><body><p>No canonical</p></body></html>"
    page = parse_page("https://example.com", html, "Test")
    assert page.canonical_url is None


# ── Section extraction ───────────────────────────────────────────────────

def test_sections_from_simple_html():
    page = parse_page("https://example.com", SIMPLE_HTML, "Test")
    headings = [s.heading for s in page.sections if s.heading]
    assert "Getting Started" in headings
    assert "Installation" in headings
    assert "Usage" in headings


def test_section_heading_levels():
    page = parse_page("https://example.com", SIMPLE_HTML, "Test")
    section_map = {s.heading: s.heading_level for s in page.sections if s.heading}
    assert section_map["Getting Started"] == 1
    assert section_map["Installation"] == 2
    assert section_map["Usage"] == 2


def test_section_content_includes_paragraphs():
    page = parse_page("https://example.com", SIMPLE_HTML, "Test")
    install_section = [s for s in page.sections if s.heading == "Installation"][0]
    assert "pip install example" in install_section.content


def test_section_content_includes_list_items():
    page = parse_page("https://example.com", SIMPLE_HTML, "Test")
    usage_section = [s for s in page.sections if s.heading == "Usage"][0]
    assert "Step one" in usage_section.content
    assert "Step two" in usage_section.content


# ── Code block preservation ──────────────────────────────────────────────

def test_code_block_preserved():
    page = parse_page("https://example.com", SIMPLE_HTML, "Test")
    assert "```python" in page.full_text or "```" in page.full_text
    assert "import example" in page.full_text


def test_inline_code_preserved():
    page = parse_page("https://example.com", SIMPLE_HTML, "Test")
    assert "pip install example" in page.full_text


# ── Table handling ───────────────────────────────────────────────────────

def test_table_to_text():
    page = parse_page("https://example.com", TABLE_HTML, "Test")
    assert "Param" in page.full_text
    assert "name" in page.full_text
    assert "str" in page.full_text


# ── Definition list ──────────────────────────────────────────────────────

def test_definition_list():
    page = parse_page("https://example.com", DL_HTML, "Test")
    assert "API" in page.full_text
    assert "Application Programming Interface" in page.full_text


# ── Blockquote ───────────────────────────────────────────────────────────

def test_blockquote():
    page = parse_page("https://example.com", BLOCKQUOTE_HTML, "Test")
    assert "This is important" in page.full_text


# ── ParsedPage fields ───────────────────────────────────────────────────

def test_parsed_page_has_doc_id():
    page = parse_page("https://example.com", SIMPLE_HTML, "Test")
    assert page.doc_id.startswith("test_")


def test_parsed_page_has_url():
    page = parse_page("https://example.com/page", SIMPLE_HTML, "Test")
    assert page.url == "https://example.com/page"


def test_parsed_page_has_source_name():
    page = parse_page("https://example.com", SIMPLE_HTML, "MySource")
    assert page.source_name == "MySource"


def test_parsed_page_has_content_hash():
    page = parse_page("https://example.com", SIMPLE_HTML, "Test")
    assert len(page.content_hash) == 32  # MD5 hex


def test_parsed_page_has_timestamp():
    page = parse_page("https://example.com", SIMPLE_HTML, "Test")
    assert page.fetch_timestamp  # non-empty ISO string


def test_parsed_page_full_text_not_empty():
    page = parse_page("https://example.com", SIMPLE_HTML, "Test")
    assert len(page.full_text) > 50


# ── Nav/chrome stripped ──────────────────────────────────────────────────

def test_nav_text_not_in_output():
    page = parse_page("https://example.com", SIMPLE_HTML, "Test")
    assert "Skip this" not in page.full_text


# ── _split_into_sections unit test ───────────────────────────────────────

def test_split_into_sections_preamble():
    text = "Preamble text.\n\n# Heading One\n\nBody one.\n\n## Heading Two\n\nBody two."
    sections = _split_into_sections(text)
    assert sections[0].heading == ""
    assert sections[0].heading_level == 0
    assert "Preamble" in sections[0].content
    assert sections[1].heading == "Heading One"
    assert sections[2].heading == "Heading Two"


def test_split_into_sections_no_headings():
    text = "Just some plain text without headings."
    sections = _split_into_sections(text)
    assert len(sections) == 1
    assert sections[0].heading_level == 0
