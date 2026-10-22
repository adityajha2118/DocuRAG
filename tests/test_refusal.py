"""
Tests for refusal behavior in src/pipeline.py (Milestone 7).

Verifies that queries with no high-scoring chunks are rejected early
without making an LLM API call.
"""

from unittest.mock import patch

import pytest

from src.pipeline import answer
from src.retrieval.hybrid import HybridResult
from src.retrieval.reranker import RerankedResult


@pytest.fixture
def mock_search_hybrid():
    with patch("src.pipeline.search_hybrid") as m:
        yield m


@pytest.fixture
def mock_rerank():
    with patch("src.pipeline.rerank") as m:
        yield m


@pytest.fixture
def mock_generate_answer():
    with patch("src.pipeline.generate_answer") as m:
        yield m


def test_refusal_when_no_results(mock_search_hybrid, mock_rerank, mock_generate_answer):
    """If the reranker returns empty (i.e. all chunks < threshold), refuse early."""
    # Simulate hybrid returning chunks, but reranker rejecting all of them
    mock_search_hybrid.return_value = [
        HybridResult(
            chunk_id="c1", doc_id="d1", source="s", url="u", title="t",
            section_heading="h", content="irrelevant content", rrf_score=0.1, dense_rank=1, bm25_rank=1
        )
    ]
    mock_rerank.return_value = []
    
    query = "What are the rules of underwater basket weaving?"
    resp = answer(query)
    
    # Assertions
    mock_generate_answer.assert_not_called()  # Critical: save LLM cost
    assert "I don't have enough information" in resp.answer
    assert len(resp.citations) == 0
    assert resp.query == query


def test_no_refusal_when_valid_results(mock_search_hybrid, mock_rerank, mock_generate_answer):
    """If reranker returns valid chunks, proceed to LLM generation."""
    mock_search_hybrid.return_value = []
    mock_rerank.return_value = [
        RerankedResult(
            chunk_id="c1", doc_id="d1", source="FastAPI", url="http://x",
            title="Intro", section_heading="Setup", content="Here is some valid content",
            rerank_score=0.5, rrf_score=0.01
        )
    ]
    mock_generate_answer.return_value = ("Here is the answer [c1].", "answer_prompt_v1")
    
    query = "How to setup FastAPI?"
    resp = answer(query)
    
    # Assertions
    mock_generate_answer.assert_called_once()
    assert resp.answer == "Here is the answer [c1]."
    assert len(resp.citations) == 1
    assert resp.citations[0].chunk_id == "c1"
