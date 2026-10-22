"""
DocuRAG — Hybrid Retrieval using Reciprocal Rank Fusion (RRF).

Combines results from Dense and BM25 retrievers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import RRF_K, RRF_TOP_K
from src.retrieval.dense import search as search_dense
from src.retrieval.bm25 import search_bm25

logger = logging.getLogger(__name__)


@dataclass
class HybridResult:
    """A combined retrieval hit from RRF."""
    chunk_id: str
    doc_id: str
    source: str
    url: str
    title: str
    section_heading: str
    content: str
    rrf_score: float
    dense_rank: int | None
    bm25_rank: int | None


def search_hybrid(query: str, top_k: int = RRF_TOP_K) -> list[HybridResult]:
    """Perform hybrid retrieval using RRF to combine Dense and BM25 results."""
    
    # 1. Get raw results (fetch plenty to ensure good overlap)
    dense_results = search_dense(query, top_k=top_k * 2)
    bm25_results = search_bm25(query, top_k=top_k * 2)
    
    # 2. Build map of chunk_id -> metadata (to preserve full info)
    chunk_metadata = {}
    for r in dense_results + bm25_results:
        if r.chunk_id not in chunk_metadata:
            chunk_metadata[r.chunk_id] = {
                "doc_id": r.doc_id,
                "source": r.source,
                "url": r.url,
                "title": r.title,
                "section_heading": r.section_heading,
                "content": r.content
            }
            
    # 3. Calculate RRF scores
    rrf_scores: dict[str, float] = {cid: 0.0 for cid in chunk_metadata}
    dense_ranks: dict[str, int] = {}
    bm25_ranks: dict[str, int] = {}
    
    # Dense ranks (1-indexed)
    for rank, res in enumerate(dense_results, start=1):
        dense_ranks[res.chunk_id] = rank
        rrf_scores[res.chunk_id] += 1.0 / (RRF_K + rank)
        
    # BM25 ranks (1-indexed)
    for rank, res in enumerate(bm25_results, start=1):
        bm25_ranks[res.chunk_id] = rank
        rrf_scores[res.chunk_id] += 1.0 / (RRF_K + rank)
        
    # 4. Sort and build final results
    sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    hybrid_results = []
    for chunk_id, rrf_score in sorted_chunks[:top_k]:
        meta = chunk_metadata[chunk_id]
        hybrid_results.append(HybridResult(
            chunk_id=chunk_id,
            doc_id=meta["doc_id"],
            source=meta["source"],
            url=meta["url"],
            title=meta["title"],
            section_heading=meta["section_heading"],
            content=meta["content"],
            rrf_score=rrf_score,
            dense_rank=dense_ranks.get(chunk_id),
            bm25_rank=bm25_ranks.get(chunk_id),
        ))
        
    logger.info("Hybrid search yielded %d results (max RRF score: %.4f)",
                len(hybrid_results), hybrid_results[0].rrf_score if hybrid_results else 0.0)
                
    return hybrid_results
