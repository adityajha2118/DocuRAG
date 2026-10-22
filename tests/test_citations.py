"""
Tests for src/generation/citations.py (Milestone 8).

Verifies extraction, validation, unmapped handling, and lexical heuristics.
"""

from src.generation.citations import extract_cited_chunk_ids, validate_and_format_citations, Citation


def test_extract_cited_chunk_ids():
    """Extracts valid bracketed chunk IDs and deduplicates them."""
    text = "Here is an answer [abc123_x] and [def456_y]. Wait, again [abc123_x]."
    ids = extract_cited_chunk_ids(text)
    assert ids == ["abc123_x", "def456_y"]


def test_validate_valid_citation():
    """A valid cited chunk that maps to metadata and passes lexical check."""
    ans = "This text talks about FastAPI paths [c1]."
    meta = {
        "c1": {"content": "FastAPI uses paths heavily.", "title": "Paths", "url": "http://x", "section_heading": "Path"}
    }
    final_ans, citations = validate_and_format_citations(ans, meta)
    
    assert final_ans == ans
    assert len(citations) == 1
    assert citations[0].chunk_id == "c1"


def test_validate_unmapped_citation():
    """If the LLM hallucinates a chunk_id not in the retrieved chunks, it's ignored.
    If no valid citations remain, the answer becomes a refusal."""
    ans = "FastAPI is cool [hallucinated]."
    meta = {
        "c1": {"content": "FastAPI uses paths heavily."}
    }
    final_ans, citations = validate_and_format_citations(ans, meta)
    
    assert "don't have enough information" in final_ans
    assert len(citations) == 0


def test_validate_fails_lexical_heuristic():
    """If a cited chunk shares zero words with the answer, it's rejected."""
    ans = "completely unrelated text about airplanes [c1]"
    meta = {
        "c1": {"content": "FastAPI uses paths heavily."}
    }
    final_ans, citations = validate_and_format_citations(ans, meta)
    
    assert "don't have enough information" in final_ans
    assert len(citations) == 0


def test_validate_preserves_refusal():
    """If the LLM outputs a refusal itself, we shouldn't overwrite it with ours,
    but we do ensure 0 citations are returned."""
    ans = "I don't have enough information in the available documentation to answer this question."
    meta = {}
    
    final_ans, citations = validate_and_format_citations(ans, meta)
    assert final_ans == ans
    assert len(citations) == 0
