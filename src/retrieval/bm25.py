"""
DocuRAG — BM25 Keyword Retrieval.

Builds an in-memory BM25 index from `data/chunks.jsonl` on first use
and provides exact-match keyword search capabilities.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass

from rank_bm25 import BM25Okapi

from src.config import CHUNKS_PATH, BM25_TOP_K

logger = logging.getLogger(__name__)


@dataclass
class BM25Result:
    """A single BM25 retrieval hit."""
    chunk_id: str
    doc_id: str
    source: str
    url: str
    title: str
    section_heading: str
    content: str
    bm25_score: float


class BM25Index:
    """Singleton wrapper for the BM25 index to avoid rebuilding per query."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._init_index()
        return cls._instance
        
    def _tokenize(self, text: str) -> list[str]:
        """Simple whitespace/punctuation tokenization for BM25."""
        return [t.lower() for t in re.findall(r"\w+", text)]
        
    def _init_index(self):
        logger.info("Initializing in-memory BM25 index from %s", CHUNKS_PATH)
        self.chunks: list[dict] = []
        if not CHUNKS_PATH.exists():
            logger.warning("Chunks file not found at %s. BM25 will be empty.", CHUNKS_PATH)
            self.bm25 = None
            return
            
        with open(CHUNKS_PATH, "r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    self.chunks.append(json.loads(line))
                    
        logger.info("Loaded %d chunks for BM25.", len(self.chunks))
        
        tokenized_corpus = [self._tokenize(c["content"]) for c in self.chunks]
        if tokenized_corpus:
            self.bm25 = BM25Okapi(tokenized_corpus)
            logger.info("BM25 index built successfully.")
        else:
            self.bm25 = None
            logger.warning("No chunks available. BM25 index is empty.")

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[BM25Result]:
        """Return the top_k BM25 matches for the query."""
        if not self.bm25 or not self.chunks:
            return []
            
        tokenized_query = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_query)
        
        # Sort by score descending
        # Enumerate to keep indices: (index, score)
        scored_indices = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        
        results = []
        for idx, score in scored_indices[:top_k]:
            if score <= 0.0:
                continue
            chunk = self.chunks[idx]
            results.append(BM25Result(
                chunk_id=chunk["chunk_id"],
                doc_id=chunk["doc_id"],
                source=chunk.get("source", ""),
                url=chunk["url"],
                title=chunk["title"],
                section_heading=chunk.get("section_heading", ""),
                content=chunk["content"],
                bm25_score=score
            ))
            
        return results


def search_bm25(query: str, top_k: int = BM25_TOP_K) -> list[BM25Result]:
    """Helper to query the global BM25 index."""
    index = BM25Index()
    return index.search(query, top_k=top_k)
