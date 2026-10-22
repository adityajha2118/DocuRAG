"""
Tests for src/ingestion/sources.py — verify source definitions are complete,
well-formed, and cover all required documentation sites.
"""

import pytest

from src.ingestion.sources import SOURCES, DocSource, get_source_by_name


# ── Required source names ────────────────────────────────────────────────

REQUIRED_NAMES = [
    "PyTorch",
    "Hugging Face Transformers",
    "scikit-learn",
    "FastAPI",
    "LangChain",
    "LangGraph",
]


def test_all_six_sources_defined():
    """Exactly 6 documentation sources must be defined."""
    assert len(SOURCES) == 6


@pytest.mark.parametrize("name", REQUIRED_NAMES)
def test_required_source_present(name: str):
    """Each required documentation source must be in the catalogue."""
    found = get_source_by_name(name)
    assert found is not None, f"Source '{name}' not found in SOURCES"


@pytest.mark.parametrize("name", REQUIRED_NAMES)
def test_source_has_valid_base_url(name: str):
    """Every source must have an https:// base URL."""
    src = get_source_by_name(name)
    assert src is not None
    assert src.base_url.startswith("https://"), f"{name} base_url must be https"


@pytest.mark.parametrize("name", REQUIRED_NAMES)
def test_source_has_non_empty_allowed_domains(name: str):
    src = get_source_by_name(name)
    assert src is not None
    assert len(src.allowed_domains) > 0, f"{name} must have ≥1 allowed domain"


@pytest.mark.parametrize("name", REQUIRED_NAMES)
def test_source_has_non_empty_allowed_prefixes(name: str):
    src = get_source_by_name(name)
    assert src is not None
    assert len(src.allowed_prefixes) > 0, f"{name} must have ≥1 allowed prefix"


@pytest.mark.parametrize("name", REQUIRED_NAMES)
def test_allowed_prefixes_start_with_slash(name: str):
    """Path prefixes must start with '/' to match URL paths."""
    src = get_source_by_name(name)
    assert src is not None
    for prefix in src.allowed_prefixes:
        assert prefix.startswith("/"), (
            f"{name} prefix '{prefix}' must start with '/'"
        )


def test_source_is_frozen_dataclass():
    """DocSource must be immutable (frozen dataclass)."""
    src = SOURCES[0]
    with pytest.raises(AttributeError):
        src.name = "SomethingElse"  # type: ignore[misc]


def test_get_source_by_name_case_insensitive():
    """Lookup should be case-insensitive."""
    assert get_source_by_name("pytorch") is not None
    assert get_source_by_name("PYTORCH") is not None
    assert get_source_by_name("PyTorch") is not None


def test_get_source_by_name_returns_none_for_unknown():
    assert get_source_by_name("nonexistent-library") is None


def test_no_duplicate_source_names():
    """Source names must be unique (case-insensitive)."""
    lower_names = [s.name.lower() for s in SOURCES]
    assert len(lower_names) == len(set(lower_names)), "Duplicate source names found"


def test_base_url_domain_in_allowed_domains():
    """The base URL's domain must be listed in allowed_domains."""
    from urllib.parse import urlparse
    for src in SOURCES:
        domain = urlparse(src.base_url).netloc
        assert domain in src.allowed_domains, (
            f"{src.name}: base_url domain '{domain}' not in allowed_domains"
        )
