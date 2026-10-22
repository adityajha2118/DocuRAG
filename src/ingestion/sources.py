"""
DocuRAG — documentation source definitions.

Each source describes one AI/ML documentation site to crawl.  Fields:
  - name:              human-readable library name
  - base_url:          entry-point URL for the crawl
  - allowed_domains:   set of domains the crawler may follow links into
  - allowed_prefixes:  URL path prefixes that are in-scope (keeps the crawl
                       focused on stable docs, not blog posts or changelogs)

This module is pure data — no I/O, no network calls — so it can be imported
and tested cheaply.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class DocSource:
    """Immutable descriptor for a single documentation source."""
    name: str
    base_url: str
    allowed_domains: frozenset[str]
    allowed_prefixes: tuple[str, ...]


# ── Source catalogue ─────────────────────────────────────────────────────

SOURCES: list[DocSource] = [
    DocSource(
        name="PyTorch",
        base_url="https://pytorch.org/docs/stable/",
        allowed_domains=frozenset({"pytorch.org"}),
        allowed_prefixes=("/docs/stable/",),
    ),
    DocSource(
        name="Hugging Face Transformers",
        base_url="https://huggingface.co/docs/transformers/",
        allowed_domains=frozenset({"huggingface.co"}),
        allowed_prefixes=("/docs/transformers/",),
    ),
    DocSource(
        name="scikit-learn",
        base_url="https://scikit-learn.org/stable/",
        allowed_domains=frozenset({"scikit-learn.org"}),
        allowed_prefixes=("/stable/",),
    ),
    DocSource(
        name="FastAPI",
        base_url="https://fastapi.tiangolo.com/",
        allowed_domains=frozenset({"fastapi.tiangolo.com"}),
        # Explicit English-only prefixes — excludes /de/, /es/, /fr/, /hi/,
        # /ja/, /ko/, /pt/, /ru/, /tr/, /uk/, /zh/, /zh-hant/ translations.
        allowed_prefixes=(
            "/tutorial/",
            "/advanced/",
            "/reference/",
            "/features/",
            "/python-types/",
            "/async/",
            "/fastapi-cli/",
            "/about/",
            "/resources/",
            "/learn/",
            "/release-notes/",
            "/deployment/",
            "/how-to/",
        ),
    ),
    DocSource(
        name="LangChain",
        base_url="https://python.langchain.com/docs/",
        allowed_domains=frozenset({"python.langchain.com"}),
        allowed_prefixes=("/docs/",),
    ),
    DocSource(
        name="LangGraph",
        base_url="https://langchain-ai.github.io/langgraph/",
        allowed_domains=frozenset({"langchain-ai.github.io"}),
        allowed_prefixes=("/langgraph/",),
    ),
]


def get_source_by_name(name: str) -> DocSource | None:
    """Look up a source by its name (case-insensitive)."""
    for src in SOURCES:
        if src.name.lower() == name.lower():
            return src
    return None
