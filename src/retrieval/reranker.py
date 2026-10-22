"""
DocuRAG — Reranking using Cross-Encoder.

Takes the top-k results from hybrid retrieval and reranks them
using a local cross-encoder model to compute relevance scores.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sentence_transformers import CrossEncoder

from src.config import RERANKER_MODEL, MIN_RERANK_SCORE
from src.retrieval.hybrid import HybridResult

logger = logging.getLogger(__name__)

# Keep a global instance of the model
_reranker_instance = None

def get_reranker() -> CrossEncoder:
    global _reranker_instance
    if _reranker_instance is None:
        logger.info("Loading cross-encoder reranker model: %s", RERANKER_MODEL)
        _reranker_instance = CrossEncoder(RERANKER_MODEL)
    return _reranker_instance


@dataclass
class RerankedResult:
    """A final retrieved chunk after cross-encoder reranking."""
    chunk_id: str
    doc_id: str
    source: str
    url: str
    title: str
    section_heading: str
    content: str
    rerank_score: float
    # Track previous scores for debugging
    rrf_score: float


def rerank(query: str, chunks: list[HybridResult], top_k: int = 5, min_score: float = MIN_RERANK_SCORE) -> list[RerankedResult]:
    """Rerank chunks against the query using a cross-encoder."""
    if not chunks:
        return []
        
    model = get_reranker()
    
    # Prepare inputs: list of (query, document) pairs
    pairs = [[query, chunk.content] for chunk in chunks]
    
    # Predict scores
    scores = model.predict(pairs)
    
    # Combine chunks with new scores
    scored_results = []
    for chunk, score in zip(chunks, scores):
        scored_results.append(RerankedResult(
            chunk_id=chunk.chunk_id,
            doc_id=chunk.doc_id,
            source=chunk.source,
            url=chunk.url,
            title=chunk.title,
            section_heading=chunk.section_heading,
            content=chunk.content,
            rerank_score=float(score),
            rrf_score=chunk.rrf_score
        ))
        
    # Sort by cross-encoder score
    scored_results.sort(key=lambda x: x.rerank_score, reverse=True)
    
    # Filter by minimum score and take top_k
    final_results = [r for r in scored_results if r.rerank_score >= min_score][:top_k]
    
    logger.info("Reranker kept %d of %d chunks (cut-off: %.2f).", 
                len(final_results), len(chunks), min_score)
                
    return final_results
