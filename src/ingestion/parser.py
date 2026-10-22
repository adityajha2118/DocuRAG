"""
DocuRAG — HTML → structured-text parser.

Converts a cleaned HTML page into a :class:`ParsedPage` with title, canonical
URL, section hierarchy, and markdown-like full text.  Every parsed page keeps
the metadata needed for downstream chunking and citation.

This module is purely transformational — no I/O, no network calls.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone

from bs4 import BeautifulSoup, NavigableString, Tag

from src.ingestion.cleaner import clean_html


# ── Data classes ─────────────────────────────────────────────────────────

@dataclass
class ParsedSection:
    """One section of a documentation page (heading + body text)."""
    heading: str
    heading_level: int          # 1-6; 0 = preamble (text before first heading)
    content: str


@dataclass
class ParsedPage:
    """Fully parsed representation of a single documentation page."""
    doc_id: str
    url: str
    canonical_url: str | None
    title: str
    source_name: str
    sections: list[ParsedSection]
    full_text: str
    fetch_timestamp: str        # ISO-8601
    content_hash: str           # MD5 hex of full_text


# ── CSS selectors tried (in priority order) for the main content area ───

_CONTENT_SELECTORS: tuple[str, ...] = (
    "article.md-content__inner",       # mkdocs-material
    ".md-content__inner",
    "div.md-content",
    "article[role='main']",
    "[role='main']",
    "main",
    "article",
    ".document .body",                 # Sphinx
    ".body",
    "#content",
    ".content",
    ".page-content",
)


# ── Internal helpers ─────────────────────────────────────────────────────

def _generate_doc_id(source_name: str, url: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", source_name.lower()).strip("_")
    url_hash = hashlib.md5(url.encode()).hexdigest()[:12]
    return f"{slug}_{url_hash}"


def _find_main_content(soup: BeautifulSoup) -> Tag:
    """Return the best-guess main-content container from *soup*."""
    for selector in _CONTENT_SELECTORS:
        el = soup.select_one(selector)
        if el and el.get_text(strip=True):
            return el
    return soup.body or soup


def _extract_title(soup: BeautifulSoup) -> str:
    """Extract the page title from <title>, <h1>, or og:title."""
    # Prefer <h1>
    h1 = soup.find("h1")
    if h1:
        text = h1.get_text(strip=True)
        if text:
            return text
    # <title> tag
    if soup.title and soup.title.string:
        return soup.title.string.strip()
    # og:title
    og = soup.find("meta", property="og:title")
    if og and og.get("content"):
        return og["content"].strip()
    return "Untitled"


def _extract_canonical_url(soup: BeautifulSoup) -> str | None:
    """Return the canonical URL if declared, else None."""
    link = soup.find("link", rel="canonical")
    if link and link.get("href"):
        return link["href"].strip()
    og = soup.find("meta", property="og:url")
    if og and og.get("content"):
        return og["content"].strip()
    return None


def _get_code_language(element: Tag) -> str:
    """Try to detect programming language from class names on a <code>/<pre>."""
    for cls in element.get("class", []):
        cls_lower = cls.lower()
        for prefix in ("language-", "highlight-", "lang-"):
            if cls_lower.startswith(prefix):
                return cls_lower[len(prefix):]
        # Sphinx uses "highlight-python" on the wrapper div
        if cls_lower in ("python", "javascript", "js", "bash", "shell",
                         "json", "yaml", "html", "css", "sql", "rust",
                         "go", "java", "cpp", "c", "ruby", "typescript"):
            return cls_lower
    return ""


def _list_to_text(element: Tag) -> str:
    """Convert a <ul> or <ol> to a text list with markers."""
    items: list[str] = []
    for i, li in enumerate(element.find_all("li", recursive=False)):
        prefix = f"{i + 1}." if element.name == "ol" else "-"
        text = li.get_text(" ", strip=True)
        if text:
            items.append(f"{prefix} {text}")
    return "\n".join(items)


def _table_to_text(table: Tag) -> str:
    """Convert a <table> to a pipe-delimited text representation."""
    rows: list[str] = []
    for tr in table.find_all("tr"):
        cells = [cell.get_text(" ", strip=True)
                 for cell in tr.find_all(["th", "td"])]
        if cells:
            rows.append(" | ".join(cells))
    if len(rows) >= 2:
        # Add a separator row after the first row (assumed header)
        sep = " | ".join("---" for _ in rows[0].split(" | "))
        rows.insert(1, sep)
    return "\n".join(rows)


def _dl_to_text(dl: Tag) -> str:
    """Convert a <dl> to readable text."""
    parts: list[str] = []
    for child in dl.children:
        if isinstance(child, Tag):
            if child.name == "dt":
                parts.append(f"**{child.get_text(' ', strip=True)}**")
            elif child.name == "dd":
                parts.append(f"  {child.get_text(' ', strip=True)}")
    return "\n".join(parts)


def _container_to_text(container: Tag) -> str:
    """Recursively convert a content container to markdown-like text.

    Handles headings, paragraphs, code blocks, lists, tables, definition
    lists, blockquotes, and generic wrapper elements (div/section).
    """
    parts: list[str] = []

    for child in container.children:
        # Bare text nodes
        if isinstance(child, NavigableString):
            text = child.strip()
            if text:
                parts.append(text)
            continue
        if not isinstance(child, Tag):
            continue

        name = child.name

        # ── Headings ──
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            text = child.get_text(" ", strip=True)
            if text:
                parts.append(f"\n{'#' * level} {text}\n")

        # ── Code blocks ──
        elif name == "pre":
            code_el = child.find("code")
            code_text = (code_el or child).get_text()
            lang = _get_code_language(code_el or child)
            parts.append(f"\n```{lang}\n{code_text.strip()}\n```\n")

        # ── Paragraphs ──
        elif name == "p":
            text = child.get_text(" ", strip=True)
            if text:
                parts.append(text)

        # ── Lists ──
        elif name in ("ul", "ol"):
            text = _list_to_text(child)
            if text:
                parts.append(text)

        # ── Tables ──
        elif name == "table":
            text = _table_to_text(child)
            if text:
                parts.append(text)

        # ── Blockquotes / admonitions ──
        elif name == "blockquote":
            text = child.get_text(" ", strip=True)
            if text:
                parts.append(f"> {text}")

        # ── Definition lists ──
        elif name == "dl":
            text = _dl_to_text(child)
            if text:
                parts.append(text)

        # ── Details / summary (collapsible sections) ──
        elif name == "details":
            summary_el = child.find("summary")
            if summary_el:
                parts.append(f"**{summary_el.get_text(' ', strip=True)}**")
            inner = _container_to_text(child)
            if inner:
                parts.append(inner)

        # ── Generic wrappers — recurse ──
        elif name in ("div", "section", "article", "main", "aside",
                       "figure", "figcaption", "span", "a"):
            inner = _container_to_text(child)
            if inner:
                parts.append(inner)

    return "\n\n".join(parts)


def _split_into_sections(text: str) -> list[ParsedSection]:
    """Split markdown-like text (with ``# Heading`` markers) into sections."""
    heading_re = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
    sections: list[ParsedSection] = []

    matches = list(heading_re.finditer(text))

    # Preamble — content before the first heading
    if matches:
        preamble = text[: matches[0].start()].strip()
    else:
        preamble = text.strip()

    if preamble:
        sections.append(ParsedSection(heading="", heading_level=0, content=preamble))

    for idx, m in enumerate(matches):
        level = len(m.group(1))
        heading = m.group(2).strip()
        start = m.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if heading or body:
            sections.append(ParsedSection(
                heading=heading, heading_level=level, content=body,
            ))

    return sections


# ── Public API ───────────────────────────────────────────────────────────

def parse_page(
    url: str,
    html: str,
    source_name: str,
    fetch_timestamp: str | None = None,
) -> ParsedPage:
    """Parse a raw HTML page into a :class:`ParsedPage`.

    Parameters
    ----------
    url : str
        The URL the page was fetched from.
    html : str
        Raw HTML content.
    source_name : str
        Human-readable source name (e.g. ``"FastAPI"``).
    fetch_timestamp : str, optional
        ISO-8601 timestamp of the fetch.  Defaults to *now* (UTC).
    """
    # Clean then re-parse via the cleaner
    cleaned_soup = clean_html(html)

    title = _extract_title(cleaned_soup)
    canonical = _extract_canonical_url(cleaned_soup)
    main_content = _find_main_content(cleaned_soup)
    full_text = _container_to_text(main_content)

    # Remove excessive blank lines
    full_text = re.sub(r"\n{3,}", "\n\n", full_text).strip()

    sections = _split_into_sections(full_text)
    content_hash = hashlib.md5(full_text.encode()).hexdigest()

    return ParsedPage(
        doc_id=_generate_doc_id(source_name, url),
        url=url,
        canonical_url=canonical,
        title=title,
        source_name=source_name,
        sections=sections,
        full_text=full_text,
        fetch_timestamp=fetch_timestamp
            or datetime.now(timezone.utc).isoformat(),
        content_hash=content_hash,
    )
