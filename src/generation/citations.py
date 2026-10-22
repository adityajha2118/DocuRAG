"""
DocuRAG — Citation mapping.

Maps chunk IDs cited by the LLM back to application-owned metadata
(title, URL, section).  We NEVER trust URLs or titles produced by the
LLM — only the chunk_id references.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass
class Citation:
    """A resolved citation linking a chunk_id to its source metadata."""
    chunk_id: str
    title: str
    url: str
    section: str
    source: str


def extract_cited_chunk_ids(answer_text: str) -> list[str]:
    """Extract chunk IDs from square-bracket citations in the LLM answer.

    E.g.  "... as shown [doc1_s0_c0_abc12345] ..." → ["doc1_s0_c0_abc12345"]
    """
    # Match anything inside square brackets that looks like a chunk_id
    raw = re.findall(r"\[([a-zA-Z0-9_]+)\]", answer_text)
    # Deduplicate while preserving order
    seen: set[str] = set()
    unique: list[str] = []
    for cid in raw:
        if cid not in seen:
            seen.add(cid)
            unique.append(cid)
    return unique


def validate_and_format_citations(
    answer_text: str,
    chunk_metadata: dict[str, dict]
) -> tuple[str, list[Citation]]:
    """Extract, validate, and resolve citations. Rejects answer if unsupported.
    
    Returns:
        tuple[str, list[Citation]]: The (possibly modified) answer and its valid citations.
    """
    cited_ids = extract_cited_chunk_ids(answer_text)
    valid_citations: list[Citation] = []
    
    # Lexical overlap heuristic
    def _lexical_overlap(ans: str, chunk_text: str) -> bool:
        ans_words = set(re.findall(r"\w+", ans.lower()))
        chunk_words = set(re.findall(r"\w+", chunk_text.lower()))
        # Check if they share at least 1 meaningful word (since answer may be short)
        overlap = ans_words.intersection(chunk_words)
        return len(overlap) >= 1
    
    for cid in cited_ids:
        meta = chunk_metadata.get(cid)
        if meta is None:
            # Chunk ID hallucinated or not retrieved for this query
            continue
            
        chunk_content = meta.get("content", "")
        if not _lexical_overlap(answer_text, chunk_content):
            # Cited chunk doesn't seem to support the text lexically
            continue
            
        valid_citations.append(Citation(
            chunk_id=cid,
            title=meta.get("title", ""),
            url=meta.get("url", ""),
            section=meta.get("section_heading", ""),
            source=meta.get("source", ""),
        ))
        
    if not valid_citations and cited_ids:
        # LLM tried to cite things but they were all invalid/hallucinated
        return "I don't have enough information in the available documentation to answer this question.", []
        
    if not valid_citations and not cited_ids:
        # Sometimes it just answers without citations. 
        if "don't have enough information" not in answer_text:
             # It answered factually without citing from evidence
             return "I don't have enough information in the available documentation to answer this question.", []
        
    return answer_text, valid_citations
