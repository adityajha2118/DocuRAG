"""
Tests for src/retrieval/bm25.py (Milestone 6).

Verifies BM25 index construction, exact-term retrieval, scoring, and
edge cases — all using synthetic in-memory data (no real corpus required).
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from src.retrieval.bm25 import BM25Index, BM25Result


# ── Helpers ──────────────────────────────────────────────────────────────

def _make_chunks_file(chunks: list[dict]) -> Path:
    """Write a temporary chunks.jsonl and return its path."""
    f = tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False, encoding="utf-8")
    for c in chunks:
        f.write(json.dumps(c) + "\n")
    f.close()
    return Path(f.name)


SAMPLE_CHUNKS = [
    {
        "chunk_id": "fastapi_c0",
        "doc_id": "fastapi_d0",
        "source": "FastAPI",
        "url": "https://fastapi.tiangolo.com/tutorial/response/",
        "title": "Response",
        "section_heading": "StreamingResponse",
        "content": "You can use StreamingResponse to stream file data. "
                   "Import StreamingResponse from fastapi.responses.",
    },
    {
        "chunk_id": "pytorch_c0",
        "doc_id": "pytorch_d0",
        "source": "PyTorch",
        "url": "https://pytorch.org/docs/stable/tensors.html",
        "title": "Tensors",
        "section_heading": "Tensor Operations",
        "content": "torch.Tensor is the central class of PyTorch. "
                   "You can create tensors with torch.tensor().",
    },
    {
        "chunk_id": "sklearn_c0",
        "doc_id": "sklearn_d0",
        "source": "scikit-learn",
        "url": "https://scikit-learn.org/stable/modules/svm.html",
        "title": "SVM",
        "section_heading": "Support Vector Machines",
        "content": "Support vector machines are supervised learning models "
                   "used for classification and regression analysis.",
    },
    {
        "chunk_id": "fastapi_c1",
        "doc_id": "fastapi_d1",
        "source": "FastAPI",
        "url": "https://fastapi.tiangolo.com/tutorial/path-params/",
        "title": "Path Parameters",
        "section_heading": "Declaring Path Parameters",
        "content": "You can declare path parameters with the same syntax "
                   "used by Python format strings.",
    },
]


@pytest.fixture(autouse=True)
def _reset_bm25_singleton():
    """Reset the BM25Index singleton between tests so each test gets a fresh index."""
    BM25Index._instance = None
    yield
    BM25Index._instance = None


@pytest.fixture
def chunks_path():
    """Write sample chunks and return the path."""
    return _make_chunks_file(SAMPLE_CHUNKS)


def test_bm25_exact_term_retrieval(chunks_path):
    """BM25 should find 'StreamingResponse' in the FastAPI chunk."""
    with patch("src.retrieval.bm25.CHUNKS_PATH", chunks_path):
        index = BM25Index()
        results = index.search("StreamingResponse", top_k=5)

    assert len(results) >= 1
    top = results[0]
    assert top.chunk_id == "fastapi_c0"
    assert top.source == "FastAPI"
    assert "StreamingResponse" in top.content


def test_bm25_returns_bm25result_type(chunks_path):
    """Every item returned should be a BM25Result dataclass."""
    with patch("src.retrieval.bm25.CHUNKS_PATH", chunks_path):
        index = BM25Index()
        results = index.search("tensor", top_k=3)

    for r in results:
        assert isinstance(r, BM25Result)
        assert isinstance(r.bm25_score, float)
        assert r.bm25_score > 0.0


def test_bm25_scores_descending(chunks_path):
    """Results should be sorted by score descending."""
    with patch("src.retrieval.bm25.CHUNKS_PATH", chunks_path):
        index = BM25Index()
        results = index.search("support vector machines classification", top_k=4)

    scores = [r.bm25_score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_bm25_respects_top_k(chunks_path):
    """BM25 should return at most top_k results."""
    with patch("src.retrieval.bm25.CHUNKS_PATH", chunks_path):
        index = BM25Index()
        results = index.search("data", top_k=2)

    assert len(results) <= 2


def test_bm25_no_results_for_gibberish(chunks_path):
    """A query with no matching terms should return zero results (all scores <= 0)."""
    with patch("src.retrieval.bm25.CHUNKS_PATH", chunks_path):
        index = BM25Index()
        results = index.search("xyzzyflurble quuxnonsense", top_k=5)

    assert len(results) == 0


def test_bm25_empty_chunks_file():
    """If chunks.jsonl is empty, BM25 should return empty results without crashing."""
    empty_path = _make_chunks_file([])
    with patch("src.retrieval.bm25.CHUNKS_PATH", empty_path):
        index = BM25Index()
        results = index.search("anything", top_k=5)

    assert results == []


def test_bm25_missing_chunks_file(tmp_path):
    """If chunks.jsonl does not exist, BM25 should gracefully return empty."""
    fake_path = tmp_path / "nonexistent.jsonl"
    with patch("src.retrieval.bm25.CHUNKS_PATH", fake_path):
        index = BM25Index()
        results = index.search("anything", top_k=5)

    assert results == []


def test_bm25_metadata_preserved(chunks_path):
    """BM25 results should carry all metadata from the original chunk."""
    with patch("src.retrieval.bm25.CHUNKS_PATH", chunks_path):
        index = BM25Index()
        results = index.search("torch.Tensor", top_k=3)

    pytorch_hits = [r for r in results if r.chunk_id == "pytorch_c0"]
    assert len(pytorch_hits) == 1
    hit = pytorch_hits[0]
    assert hit.doc_id == "pytorch_d0"
    assert hit.source == "PyTorch"
    assert hit.url == "https://pytorch.org/docs/stable/tensors.html"
    assert hit.title == "Tensors"
    assert hit.section_heading == "Tensor Operations"
