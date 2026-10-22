"""
Tests for src/retrieval/reranker.py (Milestone 7).

Verifies cross-encoder scoring, top-k selection, min-score filtering,
and edge cases — cross-encoder model is mocked.
"""

from unittest.mock import patch, MagicMock
import numpy as np

import pytest

from src.retrieval.reranker import rerank, RerankedResult
from src.retrieval.hybrid import HybridResult


# ── Helpers ──────────────────────────────────────────────────────────────

def _hybrid(chunk_id, rrf_score=0.03, content="some content", **kw):
    defaults = dict(
        chunk_id=chunk_id, doc_id=f"d_{chunk_id}", source="FastAPI",
        url=f"http://example.com/{chunk_id}", title=f"Title {chunk_id}",
        section_heading=f"Section {chunk_id}", content=content,
        rrf_score=rrf_score, dense_rank=1, bm25_rank=1,
    )
    defaults.update(kw)
    return HybridResult(**defaults)


# ── Tests ────────────────────────────────────────────────────────────────

@patch("src.retrieval.reranker.get_reranker")
def test_reranker_scores_and_sorts(mock_get_reranker):
    """Reranker should score each pair and sort by rerank_score descending."""
    mock_model = MagicMock()
    # Simulate cross-encoder scores: chunk_B > chunk_A > chunk_C
    mock_model.predict.return_value = np.array([0.5, 0.9, 0.3])
    mock_get_reranker.return_value = mock_model

    chunks = [_hybrid("A"), _hybrid("B"), _hybrid("C")]
    results = rerank("test query", chunks, top_k=5, min_score=-10.0)

    assert len(results) == 3
    assert results[0].chunk_id == "B"
    assert results[1].chunk_id == "A"
    assert results[2].chunk_id == "C"

    scores = [r.rerank_score for r in results]
    assert scores == sorted(scores, reverse=True)


@patch("src.retrieval.reranker.get_reranker")
def test_reranker_respects_top_k(mock_get_reranker):
    """Should return at most top_k results."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    mock_get_reranker.return_value = mock_model

    chunks = [_hybrid(f"c{i}") for i in range(5)]
    results = rerank("query", chunks, top_k=2, min_score=-10.0)

    assert len(results) == 2


@patch("src.retrieval.reranker.get_reranker")
def test_reranker_filters_by_min_score(mock_get_reranker):
    """Chunks below min_score should be filtered out."""
    mock_model = MagicMock()
    # Only chunk_A and chunk_B pass the 0.5 threshold
    mock_model.predict.return_value = np.array([0.8, 0.6, 0.1, -0.5])
    mock_get_reranker.return_value = mock_model

    chunks = [_hybrid(f"c{i}") for i in range(4)]
    results = rerank("query", chunks, top_k=10, min_score=0.5)

    assert len(results) == 2
    assert all(r.rerank_score >= 0.5 for r in results)


@patch("src.retrieval.reranker.get_reranker")
def test_reranker_all_below_threshold(mock_get_reranker):
    """If all chunks are below min_score, should return empty list."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([-0.5, -1.0, -2.0])
    mock_get_reranker.return_value = mock_model

    chunks = [_hybrid(f"c{i}") for i in range(3)]
    results = rerank("query", chunks, top_k=5, min_score=0.01)

    assert results == []


@patch("src.retrieval.reranker.get_reranker")
def test_reranker_preserves_metadata(mock_get_reranker):
    """RerankedResult should carry original metadata + rrf_score."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.95])
    mock_get_reranker.return_value = mock_model

    chunk = _hybrid("abc", rrf_score=0.033, source="PyTorch",
                     url="http://pytorch.org", title="Tensors")
    results = rerank("query", [chunk], top_k=5, min_score=-10.0)

    assert len(results) == 1
    r = results[0]
    assert isinstance(r, RerankedResult)
    assert r.chunk_id == "abc"
    assert r.source == "PyTorch"
    assert r.url == "http://pytorch.org"
    assert r.title == "Tensors"
    assert abs(r.rrf_score - 0.033) < 1e-6
    assert abs(r.rerank_score - 0.95) < 1e-6


@patch("src.retrieval.reranker.get_reranker")
def test_reranker_passes_correct_pairs(mock_get_reranker):
    """Reranker should form (query, chunk_content) pairs for the cross-encoder."""
    mock_model = MagicMock()
    mock_model.predict.return_value = np.array([0.5, 0.3])
    mock_get_reranker.return_value = mock_model

    chunks = [_hybrid("a", content="Alpha content"), _hybrid("b", content="Beta content")]
    rerank("my question", chunks, top_k=5, min_score=-10.0)

    call_args = mock_model.predict.call_args[0][0]
    assert call_args == [["my question", "Alpha content"], ["my question", "Beta content"]]


def test_reranker_empty_input():
    """Reranking empty input should return empty without loading the model."""
    results = rerank("query", [], top_k=5)
    assert results == []
