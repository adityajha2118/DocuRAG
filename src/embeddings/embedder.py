"""
DocuRAG — Text Embedding generation.

Uses sentence-transformers to generate dense vectors locally.
No paid APIs are used for embeddings.
"""

from __future__ import annotations
from typing import Sequence

from sentence_transformers import SentenceTransformer

from src.config import EMBEDDING_MODEL, EMBEDDING_BATCH_SIZE

# Keep a global instance of the model so we don't load it for every batch
_model_instance = None

def get_embedder() -> SentenceTransformer:
    """Return a singleton instance of the sentence transformer."""
    global _model_instance
    if _model_instance is None:
        _model_instance = SentenceTransformer(EMBEDDING_MODEL)
    return _model_instance

def embed_texts(texts: list[str] | str, batch_size: int = EMBEDDING_BATCH_SIZE) -> list[list[float]]:
    """Embed a list of text strings (or a single string) into dense vectors.
    
    Uses batching for efficient inference.
    """
    model = get_embedder()
    
    # SentenceTransformer handles batching internally if batch_size is provided to encode
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=False, convert_to_numpy=False)
    
    # Ensure list of lists of floats
    if isinstance(texts, str):
        return [float(x) for x in embeddings]
    
    return [[float(x) for x in vec] for vec in embeddings]
