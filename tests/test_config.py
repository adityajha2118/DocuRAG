"""
Tests for src/config.py — verify all required settings exist and have sane
defaults (without needing a .env file or API keys).
"""

import os
from pathlib import Path

import pytest


def test_config_imports_without_env_file():
    """config.py must import cleanly even when no .env is present."""
    from src import config  # noqa: F401


def test_embedding_model_is_local():
    """Embedding model must be the free, local sentence-transformers model."""
    from src.config import EMBEDDING_MODEL
    assert EMBEDDING_MODEL == "sentence-transformers/all-MiniLM-L6-v2"


def test_reranker_model_is_local():
    """Reranker model must be the free, local cross-encoder."""
    from src.config import RERANKER_MODEL
    assert RERANKER_MODEL == "cross-encoder/ms-marco-MiniLM-L-6-v2"


def test_chroma_path_default():
    """CHROMA_PATH should default to <project_root>/chroma_db."""
    from src.config import CHROMA_PATH, PROJECT_ROOT
    # When CHROMA_PATH env var is not set, default should point into project root
    if not os.getenv("CHROMA_PATH"):
        assert CHROMA_PATH == str(PROJECT_ROOT / "chroma_db")


def test_llm_model_has_default():
    """LLM_MODEL should default to gpt-4o-mini."""
    from src.config import LLM_MODEL
    assert LLM_MODEL == os.getenv("LLM_MODEL", "gpt-4o-mini")


def test_chunk_size_and_overlap_are_positive():
    from src.config import CHUNK_SIZE, CHUNK_OVERLAP
    assert CHUNK_SIZE > 0
    assert CHUNK_OVERLAP >= 0
    assert CHUNK_OVERLAP < CHUNK_SIZE, "Overlap must be smaller than chunk size"


def test_retrieval_top_k_values_positive():
    from src.config import DENSE_TOP_K, BM25_TOP_K, RRF_TOP_K, RERANK_TOP_K
    for name, val in [("DENSE_TOP_K", DENSE_TOP_K), ("BM25_TOP_K", BM25_TOP_K),
                      ("RRF_TOP_K", RRF_TOP_K), ("RERANK_TOP_K", RERANK_TOP_K)]:
        assert val > 0, f"{name} must be positive, got {val}"


def test_rerank_top_k_lte_rrf_top_k():
    """We rerank the RRF output, so RERANK_TOP_K ≤ RRF_TOP_K."""
    from src.config import RERANK_TOP_K, RRF_TOP_K
    assert RERANK_TOP_K <= RRF_TOP_K


def test_min_rerank_score_is_float():
    from src.config import MIN_RERANK_SCORE
    assert isinstance(MIN_RERANK_SCORE, float)


def test_crawling_constraints():
    from src.config import MAX_PAGES_PER_SOURCE, CRAWL_DELAY_SECONDS, CRAWL_MAX_RETRIES
    assert MAX_PAGES_PER_SOURCE > 0
    assert MAX_PAGES_PER_SOURCE == int(os.getenv("MAX_PAGES_PER_SOURCE", "40"))
    assert CRAWL_DELAY_SECONDS >= 1.0, "Must have ≥1s delay between requests"
    assert CRAWL_MAX_RETRIES == 2, "Hard cap of 2 retries"


def test_llm_max_tokens_set():
    """Every LLM call must have an explicit max_tokens — verify the default."""
    from src.config import LLM_MAX_TOKENS
    assert isinstance(LLM_MAX_TOKENS, int)
    assert LLM_MAX_TOKENS > 0


def test_embedding_batch_size():
    from src.config import EMBEDDING_BATCH_SIZE
    assert 1 <= EMBEDDING_BATCH_SIZE <= 256, "Batch size should be reasonable"


def test_project_paths_exist_as_path_objects():
    from src.config import PROJECT_ROOT, DATA_DIR, RAW_DIR, PROCESSED_DIR
    assert isinstance(PROJECT_ROOT, Path)
    assert isinstance(DATA_DIR, Path)
    assert isinstance(RAW_DIR, Path)
    assert isinstance(PROCESSED_DIR, Path)


def test_rrf_k_constant():
    from src.config import RRF_K
    assert RRF_K == 60
