"""
DocuRAG — structure-aware, token-aware text chunking.

Respects document → section → subsection boundaries.  Sections are kept
intact when they fit within CHUNK_SIZE.  Only sections that exceed
CHUNK_SIZE are split, with CHUNK_OVERLAP token overlap.  Very small
adjacent sections under the same parent heading are merged to avoid
excessive fragmentation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, asdict
import tiktoken

from src.config import CHUNK_SIZE, CHUNK_OVERLAP


@dataclass
class DocumentChunk:
    """A single chunk of text with its provenance metadata."""
    chunk_id: str
    doc_id: str
    source: str          # e.g. "FastAPI"
    url: str
    title: str
    section_heading: str
    content: str
    token_count: int


def get_tokenizer(model: str = "cl100k_base") -> tiktoken.Encoding:
    """Return the tiktoken encoding (defaults to OpenAI's cl100k_base)."""
    return tiktoken.get_encoding(model)


def count_tokens(text: str, tokenizer: tiktoken.Encoding | None = None) -> int:
    """Return the token count for *text*."""
    if tokenizer is None:
        tokenizer = get_tokenizer()
    return len(tokenizer.encode(text, disallowed_special=()))


def chunk_text(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
    tokenizer: tiktoken.Encoding | None = None,
) -> list[str]:
    """Split *text* into overlapping chunks of at most *chunk_size* tokens.

    Uses token-level splitting.  If the text is shorter than *chunk_size*,
    a single chunk is returned.  Empty / whitespace-only text returns [].
    """
    if not text.strip():
        return []

    if tokenizer is None:
        tokenizer = get_tokenizer()

    tokens = tokenizer.encode(text, disallowed_special=())

    if len(tokens) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(tokenizer.decode(chunk_tokens))

        if end >= len(tokens):
            break

        start += chunk_size - chunk_overlap

    return chunks


# ── Minimum merge threshold — sections shorter than this (tokens) are
#    candidates for merging with the next section under the same parent.
_MERGE_THRESHOLD = 80


def process_page_into_chunks(
    page_dict: dict,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[DocumentChunk]:
    """Convert a processed-JSON page dict into a list of :class:`DocumentChunk`.

    Strategy
    --------
    1. Walk sections in order.
    2. Merge consecutive tiny sections (< _MERGE_THRESHOLD tokens) that share
       the same parent heading to avoid excessive fragmentation.
    3. For each (possibly-merged) block that fits in *chunk_size*, emit it as
       one chunk.
    4. For blocks exceeding *chunk_size*, use :func:`chunk_text` to split with
       overlap.
    """
    doc_id = page_dict["doc_id"]
    source = page_dict.get("source_name", "")
    url = page_dict["url"]
    title = page_dict["title"]

    tokenizer = get_tokenizer()
    all_chunks: list[DocumentChunk] = []

    sections = page_dict.get("sections", [])
    if not sections:
        return all_chunks

    # --- Phase 1: build (heading, text) pairs, prepending heading --------
    blocks: list[tuple[str, str]] = []
    for sec in sections:
        heading = sec.get("heading", "")
        content = sec.get("content", "").strip()
        if not content:
            continue
        if heading and not content.startswith(heading):
            text = f"{heading}\n\n{content}"
        else:
            text = content
        blocks.append((heading, text))

    if not blocks:
        return all_chunks

    # --- Phase 2: merge tiny adjacent sections ---------------------------
    merged: list[tuple[str, str]] = []
    buf_heading, buf_text = blocks[0]
    buf_toks = count_tokens(buf_text, tokenizer)

    for heading, text in blocks[1:]:
        text_toks = count_tokens(text, tokenizer)
        combined_toks = buf_toks + text_toks

        # Merge if both are small AND combined still fits in one chunk
        if buf_toks < _MERGE_THRESHOLD and text_toks < _MERGE_THRESHOLD and combined_toks <= chunk_size:
            buf_text = buf_text + "\n\n" + text
            buf_toks = combined_toks
            # Keep the first heading as the section label
        else:
            merged.append((buf_heading, buf_text))
            buf_heading, buf_text, buf_toks = heading, text, text_toks

    merged.append((buf_heading, buf_text))

    # --- Phase 3: chunk each block and emit DocumentChunks ---------------
    for section_idx, (heading, text) in enumerate(merged):
        text_chunks = chunk_text(text, chunk_size, chunk_overlap, tokenizer=tokenizer)

        for chunk_idx, chunk_text_str in enumerate(text_chunks):
            tok_count = count_tokens(chunk_text_str, tokenizer)
            chunk_hash = hashlib.md5(chunk_text_str.encode("utf-8")).hexdigest()[:8]
            chunk_id = f"{doc_id}_s{section_idx}_c{chunk_idx}_{chunk_hash}"

            all_chunks.append(DocumentChunk(
                chunk_id=chunk_id,
                doc_id=doc_id,
                source=source,
                url=url,
                title=title,
                section_heading=heading,
                content=chunk_text_str,
                token_count=tok_count,
            ))

    return all_chunks
