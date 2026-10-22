"""Tests for src/chunking/chunker.py — Milestone 3.

Covers: token-size bounds, overlap correctness, metadata preservation,
empty / duplicate chunk handling, small-section merging.
"""

import pytest
import tiktoken

from src.chunking.chunker import (
    chunk_text,
    count_tokens,
    get_tokenizer,
    process_page_into_chunks,
    DocumentChunk,
)


# ── helpers ──────────────────────────────────────────────────────────────
@pytest.fixture
def tokenizer():
    return get_tokenizer()


def _make_page(sections, doc_id="doc1", source="TestSource",
               url="http://example.com", title="Test Page"):
    """Build a minimal page dict suitable for process_page_into_chunks."""
    return {
        "doc_id": doc_id,
        "source_name": source,
        "url": url,
        "title": title,
        "sections": sections,
    }


# ── chunk_text basics ───────────────────────────────────────────────────
class TestChunkTextBasics:
    def test_empty_string(self):
        assert chunk_text("") == []

    def test_whitespace_only(self):
        assert chunk_text("   \n\t  ") == []

    def test_short_text_single_chunk(self, tokenizer):
        text = "Hello world, this is a simple test."
        chunks = chunk_text(text, chunk_size=100, chunk_overlap=20, tokenizer=tokenizer)
        assert len(chunks) == 1
        assert chunks[0] == text


# ── Token bounds ─────────────────────────────────────────────────────────
class TestTokenBounds:
    def test_chunks_never_exceed_chunk_size(self, tokenizer):
        # Generate text that is ~1500 tokens
        text = ("FastAPI is a modern, fast web framework. " * 200).strip()
        chunk_size = 600
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=100, tokenizer=tokenizer)
        for idx, chunk in enumerate(chunks):
            toks = count_tokens(chunk, tokenizer)
            assert toks <= chunk_size, (
                f"Chunk {idx} has {toks} tokens, exceeds limit {chunk_size}"
            )

    def test_last_chunk_may_be_smaller(self, tokenizer):
        text = ("word " * 300).strip()  # ~300 tokens
        chunks = chunk_text(text, chunk_size=200, chunk_overlap=50, tokenizer=tokenizer)
        assert len(chunks) >= 2
        last_toks = count_tokens(chunks[-1], tokenizer)
        assert last_toks <= 200


# ── Overlap correctness ─────────────────────────────────────────────────
class TestOverlap:
    def test_adjacent_chunks_share_tokens(self, tokenizer):
        text = ("word " * 500).strip()
        chunk_size = 200
        overlap = 50
        chunks = chunk_text(text, chunk_size=chunk_size, chunk_overlap=overlap, tokenizer=tokenizer)

        assert len(chunks) >= 3
        for i in range(len(chunks) - 1):
            end_of_a = chunks[i]
            start_of_b = chunks[i + 1]
            # The end of chunk A should appear at the start of chunk B
            # We check token-level overlap
            toks_a = tokenizer.encode(end_of_a, disallowed_special=())
            toks_b = tokenizer.encode(start_of_b, disallowed_special=())
            # Last `overlap` tokens of A should equal first `overlap` tokens of B
            assert toks_a[-overlap:] == toks_b[:overlap], (
                f"Chunks {i} and {i+1} don't share the expected {overlap}-token overlap"
            )


# ── Metadata preservation ───────────────────────────────────────────────
class TestMetadataPreservation:
    def test_chunk_carries_all_metadata(self):
        page = _make_page(
            sections=[{"heading": "Intro", "content": "Hello world."}],
            doc_id="myid",
            source="FastAPI",
            url="http://example.com/intro",
            title="My Page",
        )
        chunks = process_page_into_chunks(page)
        assert len(chunks) >= 1
        c = chunks[0]
        assert c.doc_id == "myid"
        assert c.source == "FastAPI"
        assert c.url == "http://example.com/intro"
        assert c.title == "My Page"
        assert c.section_heading == "Intro"
        assert c.token_count > 0
        assert c.chunk_id.startswith("myid_s0_c0_")

    def test_section_heading_prepended(self):
        page = _make_page(
            sections=[{"heading": "Setup", "content": "Install the package."}],
        )
        chunks = process_page_into_chunks(page)
        assert "Setup" in chunks[0].content
        assert "Install the package." in chunks[0].content


# ── Empty / duplicate handling ───────────────────────────────────────────
class TestEdgeCases:
    def test_empty_sections_skipped(self):
        page = _make_page(sections=[
            {"heading": "H1", "content": ""},
            {"heading": "H2", "content": "   "},
            {"heading": "H3", "content": "Real content."},
        ])
        chunks = process_page_into_chunks(page)
        assert len(chunks) == 1
        assert "Real content." in chunks[0].content

    def test_no_sections(self):
        page = _make_page(sections=[])
        assert process_page_into_chunks(page) == []

    def test_unique_chunk_ids(self):
        page = _make_page(sections=[
            {"heading": "A", "content": ("word " * 400).strip()},
            {"heading": "B", "content": ("other " * 400).strip()},
        ])
        chunks = process_page_into_chunks(page, chunk_size=200, chunk_overlap=50)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected"


# ── Small-section merging / fragmentation ────────────────────────────────
class TestSmallSectionMerging:
    def test_tiny_sections_merged(self, tokenizer):
        """Consecutive very-small sections should be merged into one chunk."""
        page = _make_page(sections=[
            {"heading": "Note1", "content": "Short note A."},
            {"heading": "Note2", "content": "Short note B."},
            {"heading": "Note3", "content": "Short note C."},
        ])
        chunks = process_page_into_chunks(page, chunk_size=600, chunk_overlap=100)
        # All three are tiny; they should be merged into 1 chunk
        assert len(chunks) == 1
        assert "Short note A." in chunks[0].content
        assert "Short note C." in chunks[0].content

    def test_large_section_not_merged(self, tokenizer):
        """A section exceeding the merge threshold stays separate."""
        big = ("word " * 200).strip()  # well above 80-token merge threshold
        page = _make_page(sections=[
            {"heading": "Small", "content": "Tiny."},
            {"heading": "Big", "content": big},
        ])
        chunks = process_page_into_chunks(page, chunk_size=600, chunk_overlap=100)
        # The small section and the big one should NOT be merged
        assert len(chunks) == 2


# ── Token-count field ───────────────────────────────────────────────────
class TestTokenCountField:
    def test_token_count_accurate(self, tokenizer):
        page = _make_page(sections=[
            {"heading": "H", "content": ("word " * 100).strip()},
        ])
        chunks = process_page_into_chunks(page)
        for c in chunks:
            expected = count_tokens(c.content, tokenizer)
            assert c.token_count == expected
