"""
DocuRAG — Dense retrieval via ChromaDB.

Queries the persistent Chroma collection for the top-k most similar
chunks given a text query.  Embedding is done locally using the same
model that built the index.
"""

from __future__ import annotations

from dataclasses import dataclass

import chromadb

from src.config import CHROMA_PATH, CHROMA_COLLECTION_NAME, DENSE_TOP_K
from src.embeddings.embedder import embed_texts


@dataclass
class DenseResult:
    """A single dense-retrieval hit."""
    chunk_id: str
    doc_id: str
    source: str
    url: str
    title: str
    section_heading: str
    content: str
    dense_score: float   # Chroma distance → converted to similarity


def _get_collection() -> chromadb.Collection:
    """Return the persistent Chroma collection (read-only path)."""
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    return client.get_collection(name=CHROMA_COLLECTION_NAME)


def search(query: str, top_k: int = DENSE_TOP_K) -> list[DenseResult]:
    """Return the *top_k* nearest chunks for *query*.

    Chroma returns L2 distances by default; we convert to a similarity
    score via  sim = 1 / (1 + distance).
    """
    collection = _get_collection()

    # Embed the query using the same local model
    query_embedding = embed_texts(query)  # single string → flat list

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(top_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    hits: list[DenseResult] = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    metas = results["metadatas"][0]
    dists = results["distances"][0]

    for cid, doc, meta, dist in zip(ids, docs, metas, dists):
        hits.append(DenseResult(
            chunk_id=cid,
            doc_id=meta.get("doc_id", ""),
            source=meta.get("source", ""),
            url=meta.get("url", ""),
            title=meta.get("title", ""),
            section_heading=meta.get("section_heading", ""),
            content=doc,
            dense_score=1.0 / (1.0 + dist),
        ))

    return hits
