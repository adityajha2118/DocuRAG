"""
Tests for src/retrieval/hybrid.py (Milestone 6).

Verifies RRF fusion, duplicate merging, score computation, rank tracking,
and edge cases — fully mocked, no real index needed.
"""

from unittest.mock import patch, MagicMock

import pytest

from src.retrieval.hybrid import search_hybrid, HybridResult
from src.retrieval.dense import DenseResult
from src.retrieval.bm25 import BM25Result


# ── Helpers ──────────────────────────────────────────────────────────────

def _dense(chunk_id, score=0.9, **kw):
    defaults = dict(
        chunk_id=chunk_id, doc_id=f"d_{chunk_id}", source="FastAPI",
        url=f"http://example.com/{chunk_id}", title=f"Title {chunk_id}",
        section_heading=f"Section {chunk_id}", content=f"Content of {chunk_id}",
        dense_score=score,
    )
    defaults.update(kw)
    return DenseResult(**defaults)


def _bm25(chunk_id, score=5.0, **kw):
    defaults = dict(
        chunk_id=chunk_id, doc_id=f"d_{chunk_id}", source="FastAPI",
        url=f"http://example.com/{chunk_id}", title=f"Title {chunk_id}",
        section_heading=f"Section {chunk_id}", content=f"Content of {chunk_id}",
        bm25_score=score,
    )
    defaults.update(kw)
    return BM25Result(**defaults)


# ── Tests ────────────────────────────────────────────────────────────────

@patch("src.retrieval.hybrid.search_bm25")
@patch("src.retrieval.hybrid.search_dense")
def test_rrf_score_formula(mock_dense, mock_bm25):
    """RRF score should be 1/(k+rank) summed across retrieval systems."""
    # chunk_A is rank 1 in both, chunk_B is rank 2 in both
    mock_dense.return_value = [_dense("chunk_A"), _dense("chunk_B", score=0.5)]
    mock_bm25.return_value = [_bm25("chunk_A"), _bm25("chunk_B", score=2.0)]

    with patch("src.retrieval.hybrid.RRF_K", 60):
        results = search_hybrid("test query", top_k=10)

    # chunk_A: 1/(60+1) + 1/(60+1) = 2/61 ≈ 0.03279
    a = [r for r in results if r.chunk_id == "chunk_A"][0]
    expected_a = 2.0 / 61.0
    assert abs(a.rrf_score - expected_a) < 1e-6

    # chunk_B: 1/(60+2) + 1/(60+2) = 2/62 ≈ 0.03226
    b = [r for r in results if r.chunk_id == "chunk_B"][0]
    expected_b = 2.0 / 62.0
    assert abs(b.rrf_score - expected_b) < 1e-6


@patch("src.retrieval.hybrid.search_bm25")
@patch("src.retrieval.hybrid.search_dense")
def test_rrf_duplicate_merging(mock_dense, mock_bm25):
    """If the same chunk_id appears in both retrievers, it should appear only once in results."""
    mock_dense.return_value = [_dense("shared"), _dense("dense_only")]
    mock_bm25.return_value = [_bm25("shared"), _bm25("bm25_only")]

    results = search_hybrid("query", top_k=10)

    ids = [r.chunk_id for r in results]
    assert len(ids) == len(set(ids)), "Duplicate chunk_ids in results!"
    assert "shared" in ids
    assert "dense_only" in ids
    assert "bm25_only" in ids


@patch("src.retrieval.hybrid.search_bm25")
@patch("src.retrieval.hybrid.search_dense")
def test_rrf_tracks_ranks(mock_dense, mock_bm25):
    """HybridResult should track dense_rank and bm25_rank for debugging."""
    mock_dense.return_value = [_dense("A"), _dense("B")]
    mock_bm25.return_value = [_bm25("B"), _bm25("C")]

    results = search_hybrid("query", top_k=10)
    result_map = {r.chunk_id: r for r in results}

    # A: rank 1 in dense, not in bm25
    assert result_map["A"].dense_rank == 1
    assert result_map["A"].bm25_rank is None

    # B: rank 2 in dense, rank 1 in bm25
    assert result_map["B"].dense_rank == 2
    assert result_map["B"].bm25_rank == 1

    # C: not in dense, rank 2 in bm25
    assert result_map["C"].dense_rank is None
    assert result_map["C"].bm25_rank == 2


@patch("src.retrieval.hybrid.search_bm25")
@patch("src.retrieval.hybrid.search_dense")
def test_rrf_sorted_descending(mock_dense, mock_bm25):
    """Results should be sorted by RRF score descending."""
    mock_dense.return_value = [_dense("A"), _dense("B"), _dense("C")]
    mock_bm25.return_value = [_bm25("A"), _bm25("C"), _bm25("B")]

    results = search_hybrid("query", top_k=10)
    scores = [r.rrf_score for r in results]
    assert scores == sorted(scores, reverse=True)


@patch("src.retrieval.hybrid.search_bm25")
@patch("src.retrieval.hybrid.search_dense")
def test_rrf_respects_top_k(mock_dense, mock_bm25):
    """Should return at most top_k fused results."""
    mock_dense.return_value = [_dense(f"d{i}") for i in range(10)]
    mock_bm25.return_value = [_bm25(f"b{i}") for i in range(10)]

    results = search_hybrid("query", top_k=3)
    assert len(results) <= 3


@patch("src.retrieval.hybrid.search_bm25")
@patch("src.retrieval.hybrid.search_dense")
def test_rrf_empty_retrievers(mock_dense, mock_bm25):
    """If both retrievers return empty, hybrid should return empty."""
    mock_dense.return_value = []
    mock_bm25.return_value = []

    results = search_hybrid("query", top_k=10)
    assert results == []


@patch("src.retrieval.hybrid.search_bm25")
@patch("src.retrieval.hybrid.search_dense")
def test_rrf_one_retriever_empty(mock_dense, mock_bm25):
    """If one retriever is empty, scores should still be correct from the other."""
    mock_dense.return_value = [_dense("only_dense")]
    mock_bm25.return_value = []

    with patch("src.retrieval.hybrid.RRF_K", 60):
        results = search_hybrid("query", top_k=10)

    assert len(results) == 1
    r = results[0]
    assert r.chunk_id == "only_dense"
    expected_score = 1.0 / (60 + 1)
    assert abs(r.rrf_score - expected_score) < 1e-6
    assert r.dense_rank == 1
    assert r.bm25_rank is None


@patch("src.retrieval.hybrid.search_bm25")
@patch("src.retrieval.hybrid.search_dense")
def test_hybrid_result_type(mock_dense, mock_bm25):
    """Every item returned should be a HybridResult dataclass."""
    mock_dense.return_value = [_dense("x")]
    mock_bm25.return_value = [_bm25("x")]

    results = search_hybrid("query", top_k=5)
    for r in results:
        assert isinstance(r, HybridResult)
