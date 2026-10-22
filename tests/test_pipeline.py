"""
Tests for src/pipeline.py (Milestone 7).

Verifies the end-to-end pipeline, mocking the LLM generator and retrievers.
"""

from unittest.mock import patch, MagicMock
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


def test_pipeline_refusal_when_no_reranked_results(mock_search_hybrid, mock_rerank, mock_generate_answer):
    """If the reranker filters out all results, the pipeline should refuse immediately."""
    # Hybrid returns something, but reranker filters it all out (empty list)
    mock_search_hybrid.return_value = [
        HybridResult(
            chunk_id="c1", doc_id="d1", source="s", url="u", title="t",
            section_heading="h", content="c", rrf_score=0.1, dense_rank=1, bm25_rank=1
        )
    ]
    mock_rerank.return_value = []
    
    resp = answer("What is XYZ?")
    
    # Should not call the LLM
    mock_generate_answer.assert_not_called()
    assert "I don't have enough information" in resp.answer
    assert len(resp.citations) == 0


def test_pipeline_success_and_citations(mock_search_hybrid, mock_rerank, mock_generate_answer):
    """Pipeline should pass chunks to generator and resolve citations correctly."""
    
    mock_search_hybrid.return_value = [] # Content doesn't matter, mock_rerank is what pipeline uses
    
    mock_rerank.return_value = [
        RerankedResult(
            chunk_id="chunk1", doc_id="d1", source="FastAPI", url="http://x",
            title="Intro", section_heading="Setup", content="content 1",
            rerank_score=0.9, rrf_score=0.01
        ),
        RerankedResult(
            chunk_id="chunk2", doc_id="d2", source="FastAPI", url="http://y",
            title="Advanced", section_heading="Auth", content="content 2",
            rerank_score=0.8, rrf_score=0.01
        )
    ]
    
    # Mock LLM to return an answer containing citations
    mock_generate_answer.return_value = "FastAPI setup is easy [chunk1]. It supports Auth [chunk2]."
    
    resp = answer("How to setup FastAPI?")
    
    mock_generate_answer.assert_called_once()
    assert "FastAPI setup is easy" in resp.answer
    assert len(resp.citations) == 2
    
    c1 = [c for c in resp.citations if c.chunk_id == "chunk1"][0]
    assert c1.title == "Intro"
    assert c1.url == "http://x"
    assert c1.source == "FastAPI"
