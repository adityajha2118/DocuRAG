"""
DocuRAG — centralised configuration.

All tuneable parameters live here. Values can be overridden via environment
variables (loaded from .env by python-dotenv).  Application code imports
from this module — never reads os.environ directly.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # reads .env if present; no-op otherwise

# ── Project paths ────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent  # repo root
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_PATH = DATA_DIR / "metadata.jsonl"
CHUNKS_PATH = DATA_DIR / "chunks.jsonl"

# ── Embedding (LOCAL — never call a paid API) ────────────────────────────
EMBEDDING_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_BATCH_SIZE: int = int(os.getenv("EMBEDDING_BATCH_SIZE", "64"))

# ── Reranker (LOCAL cross-encoder — never call a paid API) ───────────────
RERANKER_MODEL: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# ── ChromaDB (persistent, local) ────────────────────────────────────────
CHROMA_PATH: str = os.getenv("CHROMA_PATH", str(PROJECT_ROOT / "chroma_db"))
CHROMA_COLLECTION_NAME: str = "ai_ml_documentation"

# ── LLM generation (paid API — used ONLY for answer gen & RAGAS eval) ───
LLM_MODEL: str = os.getenv("LLM_MODEL", "gpt-4o-mini")
LLM_API_KEY: str = os.getenv("OPENAI_API_KEY", "") or os.getenv("ANTHROPIC_API_KEY", "")
LLM_BASE_URL: str | None = os.getenv("OPENAI_BASE_URL", None)
LLM_MAX_TOKENS: int = int(os.getenv("LLM_MAX_TOKENS", "1024"))

# ── Chunking ─────────────────────────────────────────────────────────────
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "600"))        # tokens (target 500-800)
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "100"))  # tokens (~100 overlap)

# ── Retrieval ────────────────────────────────────────────────────────────
DENSE_TOP_K: int = int(os.getenv("DENSE_TOP_K", "20"))
BM25_TOP_K: int = int(os.getenv("BM25_TOP_K", "20"))
RRF_TOP_K: int = int(os.getenv("RRF_TOP_K", "20"))
RRF_K: int = 60  # reciprocal rank fusion constant

# ── Reranking ────────────────────────────────────────────────────────────
RERANK_TOP_K: int = int(os.getenv("RERANK_TOP_K", "5"))
MIN_RERANK_SCORE: float = float(os.getenv("MIN_RERANK_SCORE", "0.01"))

# ── Crawling ─────────────────────────────────────────────────────────────
MAX_PAGES_PER_SOURCE: int = int(os.getenv("MAX_PAGES_PER_SOURCE", "40"))
CRAWL_DELAY_SECONDS: float = float(os.getenv("CRAWL_DELAY_SECONDS", "1.0"))
CRAWL_TIMEOUT_SECONDS: int = int(os.getenv("CRAWL_TIMEOUT_SECONDS", "15"))
CRAWL_MAX_RETRIES: int = 2  # exponential backoff, hard cap

# ── Langfuse Observability ───────────────────────────────────────────────
LANGFUSE_ENABLED: bool = os.getenv("LANGFUSE_ENABLED", "true").lower() in ("1", "true", "yes")
LANGFUSE_HOST: str = os.getenv("LANGFUSE_HOST", "http://localhost:3000")
LANGFUSE_PUBLIC_KEY: str = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY: str = os.getenv("LANGFUSE_SECRET_KEY", "")

# ── Metrics Store ────────────────────────────────────────────────────────
METRICS_DB_PATH: str = os.getenv("METRICS_DB_PATH", str(PROJECT_ROOT / "data" / "metrics.db"))

# ── Token Pricing (USD per 1K tokens) ────────────────────────────────────
# Add new models here as needed. Costs are looked up by LLM_MODEL at runtime.
# Format: { "model_name": {"input": cost_per_1k, "output": cost_per_1k} }
TOKEN_PRICING: dict[str, dict[str, float]] = {
    # NVIDIA NIM hosted models
    "deepseek-ai/deepseek-v4.1-flash":  {"input": 0.00,  "output": 0.00},   # free tier
    "meta/llama-3.1-70b-instruct":      {"input": 0.00,  "output": 0.00},   # free tier
    # OpenAI models (for reference / future use)
    "gpt-4o-mini":                      {"input": 0.15,  "output": 0.60},
    "gpt-4o":                           {"input": 2.50,  "output": 10.00},
    # Default fallback for unknown models
    "_default":                         {"input": 0.00,  "output": 0.00},
}
