"""
DocuRAG — HTML cleaner.

Strips navigation, sidebars, footers, cookie banners, scripts, styles, and
other non-content elements from raw HTML while preserving headings, prose,
lists, and code blocks.

This module is purely transformational (HTML in → cleaned HTML out) — no I/O.
"""

from __future__ import annotations

from bs4 import BeautifulSoup, Comment, Tag


# ── Tags to remove entirely (element + all children) ────────────────────
_STRIP_TAGS: frozenset[str] = frozenset({
    "script", "style", "noscript", "svg", "iframe", "object", "embed",
    "nav", "footer",
})

# ── Class-name substrings that indicate non-content chrome ──────────────
_REMOVE_CLASS_SUBSTRINGS: tuple[str, ...] = (
    # Generic layout chrome
    "sidebar", "side-bar", "sidenav",
    "navbar", "nav-bar", "navigation",
    "footer", "site-footer",
    "site-header", "page-header",
    "toc ", "toc-", "table-of-contents", "tocnav",
    "breadcrumb",
    "cookie", "consent", "banner",
    "toolbar", "toolbox",
    "searchbox", "search-box", "search-form", "search-modal",
    "social", "share",
    "modal", "dialog", "popup",
    "announcement", "alert-banner",
    "skip-link",
    # mkdocs-material
    "md-header", "md-footer", "md-sidebar", "md-tabs",
    "md-top", "md-dialog", "md-source", "md-search",
    # Sphinx / ReadTheDocs
    "sphinxsidebar", "related", "headerlink",
    "rst-versions", "injected",
    # Docusaurus
    "docSidebarContainer", "pagination-nav", "theme-doc-footer",
)

# ── ID substrings that indicate non-content chrome ──────────────────────
_REMOVE_ID_SUBSTRINGS: tuple[str, ...] = (
    "sidebar", "sidenav",
    "navbar", "navigation",
    "footer",
    "toc", "table-of-contents",
    "cookie", "consent", "banner",
    "search-modal", "searchbox",
)

# ── ARIA roles that mark non-content landmarks ──────────────────────────
_REMOVE_ROLES: frozenset[str] = frozenset({
    "navigation", "banner", "contentinfo", "search", "complementary",
})


# ── Helpers ──────────────────────────────────────────────────────────────

def _should_remove(tag: Tag) -> bool:
    """Return True if *tag* looks like non-content chrome."""
    # Check CSS classes
    classes = tag.get("class", [])
    if isinstance(classes, str):
        classes = [classes]
    class_str = " ".join(classes).lower()
    for pat in _REMOVE_CLASS_SUBSTRINGS:
        if pat in class_str:
            return True

    # Check ID
    tag_id = (tag.get("id") or "").lower()
    for pat in _REMOVE_ID_SUBSTRINGS:
        if pat in tag_id:
            return True

    # Check ARIA role
    if (tag.get("role") or "").lower() in _REMOVE_ROLES:
        return True

    return False


# ── Public API ───────────────────────────────────────────────────────────

def clean_html(html: str) -> BeautifulSoup:
    """Remove non-content elements from *html*.

    Returns a :class:`BeautifulSoup` tree containing only content-relevant
    elements (headings, paragraphs, lists, code blocks, tables, etc.).
    """
    soup = BeautifulSoup(html, "lxml")

    # 1. HTML comments
    for comment in soup.find_all(string=lambda t: isinstance(t, Comment)):
        comment.extract()

    # 2. Blacklisted tags (script, style, nav, footer, …)
    for tag_name in _STRIP_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # 3. <header> — only remove when it's a site-level header
    for header in soup.find_all("header"):
        is_site_header = (
            (header.parent and header.parent.name in ("body", "[document]"))
            or _should_remove(header)
        )
        if is_site_header:
            header.decompose()

    # 4. Elements matching class / id / role patterns
    to_remove: list[Tag] = []
    for tag in soup.find_all(True):
        if _should_remove(tag):
            to_remove.append(tag)
    for tag in to_remove:
        if tag.parent is not None:          # may already be gone (nested)
            tag.decompose()

    # 5. Unwrap residual inline styling tags (preserve their text content).
    #    Documentation sites (especially mkdocs-material) embed <font>,
    #    <span style="…">, and <u style="…"> in code blocks for coloured
    #    terminal output.  We strip the tags but keep the text.
    for tag_name in ("font", ):
        for tag in soup.find_all(tag_name):
            tag.unwrap()
    for tag in soup.find_all("span", style=True):
        tag.unwrap()
    for tag in soup.find_all("u", style=True):
        tag.unwrap()

    return soup

