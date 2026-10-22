"""
DocuRAG — End-to-end question answering pipeline (Milestone 6/7 + 13/14 Observability).

Retrieves relevant chunks via hybrid search (Dense + BM25), reranks them
with a cross-encoder, generates an answer via an LLM, and maps the cited
chunk IDs back to source citations.

Every call is wrapped in a Langfuse trace with nested spans per stage,
and per-request metrics are recorded to SQLite.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from src.retrieval.hybrid import search_hybrid, HybridResult
from src.retrieval.reranker import rerank, RerankedResult
from src.generation.generator import generate_answer
from src.generation.citations import Citation
from src.tracing import create_trace, timed_span, flush as flush_traces
from src.observability.metrics import record_request

logger = logging.getLogger(__name__)


@dataclass
class AnswerResponse:
    """The final response sent to the user."""
    query: str
    answer: str
    citations: list[Citation]
    retrieved_chunks: list[RerankedResult]


def answer(query: str, use_cache: bool = True) -> AnswerResponse:
    """Answer a user query by retrieving evidence, reranking, and calling the LLM.

    If top reranked results have very low scores, we fast-fail and refuse
    rather than hallucinate.
    """
    logger.info("Pipeline started for query: %s", query)
    pipeline_start = time.perf_counter()

    # ── Create Langfuse trace for this request ───────────────────────
    trace = create_trace(
        name="docurag.answer",
        input={"query": query, "use_cache": use_cache},
        metadata={"pipeline_version": "v1"},
    )

    stage_latencies = {}  # collect per-stage timings for metrics DB

    # ── 1. Dense Retrieval ───────────────────────────────────────────
    _t = time.perf_counter()
    with timed_span(trace, name="dense_retrieval", input={"query": query}) as ctx:
        from src.retrieval.dense import search as search_dense
        from src.config import DENSE_TOP_K
        dense_results = search_dense(query, top_k=DENSE_TOP_K * 2)
        ctx["output"] = {
            "n_results": len(dense_results),
            "chunk_ids": [r.chunk_id for r in dense_results[:10]],
        }
    stage_latencies["dense_retrieval"] = int((time.perf_counter() - _t) * 1000)

    # ── 2. BM25 Retrieval ────────────────────────────────────────────
    _t = time.perf_counter()
    with timed_span(trace, name="bm25_retrieval", input={"query": query}) as ctx:
        from src.retrieval.bm25 import search_bm25
        from src.config import BM25_TOP_K
        bm25_results = search_bm25(query, top_k=BM25_TOP_K * 2)
        ctx["output"] = {
            "n_results": len(bm25_results),
            "chunk_ids": [r.chunk_id for r in bm25_results[:10]],
        }
    stage_latencies["bm25_retrieval"] = int((time.perf_counter() - _t) * 1000)

    # ── 3. Hybrid Fusion (RRF) ───────────────────────────────────────
    _t = time.perf_counter()
    with timed_span(trace, name="hybrid_fusion",
                    input={"dense_count": len(dense_results), "bm25_count": len(bm25_results)}) as ctx:
        hybrid_results = search_hybrid(query)
        ctx["output"] = {
            "n_results": len(hybrid_results),
            "top_rrf_score": hybrid_results[0].rrf_score if hybrid_results else 0.0,
            "chunk_ids_with_scores": [
                {"id": r.chunk_id, "rrf": round(r.rrf_score, 4)}
                for r in hybrid_results[:10]
            ],
        }
    stage_latencies["hybrid_fusion"] = int((time.perf_counter() - _t) * 1000)

    # ── 4. Reranking ─────────────────────────────────────────────────
    _t = time.perf_counter()
    with timed_span(trace, name="reranking",
                    input={"n_candidates": len(hybrid_results)}) as ctx:
        reranked_results = rerank(query, hybrid_results)
        ctx["output"] = {
            "n_kept": len(reranked_results),
            "n_filtered": len(hybrid_results) - len(reranked_results),
            "chunk_ids_with_scores": [
                {"id": r.chunk_id, "rerank_score": round(r.rerank_score, 4)}
                for r in reranked_results
            ],
        }
    stage_latencies["reranking"] = int((time.perf_counter() - _t) * 1000)

    # ── 5. Evidence Gate Decision ────────────────────────────────────
    refused = not reranked_results
    _t = time.perf_counter()
    with timed_span(trace, name="evidence_gate",
                    input={"n_reranked": len(reranked_results)}) as ctx:
        ctx["output"] = {
            "refused": refused,
            "reason": "no_results_above_threshold" if refused else "passed",
        }
    stage_latencies["evidence_gate"] = int((time.perf_counter() - _t) * 1000)

    if refused:
        logger.warning("No results passed the reranker threshold. Refusing query.")
        total_latency_ms = int((time.perf_counter() - pipeline_start) * 1000)
        trace.update(
            output={
                "answer": "I don't have enough information...",
                "refused": True,
                "citations": [],
                "latency_ms": total_latency_ms,
            }
        )
        # Record metrics for refused requests
        record_request(
            query=query,
            total_latency_ms=total_latency_ms,
            stage_latencies=stage_latencies,
            input_tokens=0,
            output_tokens=0,
            refused=True,
        )
        return AnswerResponse(
            query=query,
            answer="I don't have enough information in the available documentation to answer this question.",
            citations=[],
            retrieved_chunks=[],
        )
        
    # ── 6. Format chunks for generator ───────────────────────────────
    chunk_dicts = []
    chunk_meta_map = {}
    for r in reranked_results:
        c_dict = {
            "chunk_id": r.chunk_id,
            "content": r.content,
            "title": r.title,
            "section_heading": r.section_heading,
            "url": r.url,
            "source": r.source,
        }
        chunk_dicts.append(c_dict)
        chunk_meta_map[r.chunk_id] = c_dict
        
    # ── 7. LLM Generation ───────────────────────────────────────────
    _t = time.perf_counter()
    with timed_span(trace, name="llm_generation",
                    input={"n_chunks": len(chunk_dicts), "use_cache": use_cache},
                    metadata={"model": "from_config"}) as ctx:
        answer_text, prompt_version = generate_answer(query, chunk_dicts, use_cache=use_cache)
        ctx["output"] = {
            "prompt_version": prompt_version,
            "answer_length": len(answer_text),
            "cache_hit": use_cache,
        }
    stage_latencies["llm_generation"] = int((time.perf_counter() - _t) * 1000)

    # ── 8. Citation Validation ───────────────────────────────────────
    _t = time.perf_counter()
    with timed_span(trace, name="citation_validation",
                    input={"raw_answer_length": len(answer_text)}) as ctx:
        from src.generation.citations import validate_and_format_citations
        answer_text, citations = validate_and_format_citations(answer_text, chunk_meta_map)
        ctx["output"] = {
            "n_valid_citations": len(citations),
            "cited_chunk_ids": [c.chunk_id for c in citations],
            "answer_modified": "don't have enough information" in answer_text,
        }
    stage_latencies["citation_validation"] = int((time.perf_counter() - _t) * 1000)

    # ── Finalise trace ───────────────────────────────────────────────
    total_latency_ms = int((time.perf_counter() - pipeline_start) * 1000)

    trace.update(
        output={
            "answer": answer_text[:500],  # truncate for lean payloads
            "refused": len(citations) == 0,
            "n_citations": len(citations),
            "prompt_version": prompt_version,
            "total_latency_ms": total_latency_ms,
        },
        metadata={
            "pipeline_version": "v1",
            "prompt_version": prompt_version,
        },
    )

    # ── Record metrics to SQLite ─────────────────────────────────────
    citations_valid = len(citations) > 0
    # Simple token estimation: 1 token ~= 4 chars
    input_chars = len(query) + sum(len(c["content"]) for c in chunk_dicts)
    input_tokens = input_chars // 4
    output_tokens = len(answer_text) // 4
    
    record_request(
        query=query,
        total_latency_ms=total_latency_ms,
        stage_latencies=stage_latencies,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        prompt_version=prompt_version,
        n_citations=len(citations),
        citations_valid=citations_valid,
        refused=False,
    )

    # Simple structured logging (preserved from original)
    logger.info(
        "QUERY_LOG: query=%r prompt_version=%r retrieved=%d reranked=%d citations=%d refused=%s latency_ms=%d",
        query, prompt_version, len(hybrid_results), len(reranked_results), len(citations), len(citations) == 0, total_latency_ms
    )
    
    return AnswerResponse(
        query=query,
        answer=answer_text,
        citations=citations,
        retrieved_chunks=reranked_results,
    )
