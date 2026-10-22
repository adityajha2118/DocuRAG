"""
DocuRAG — Langfuse tracing wrapper (Milestone 13).

Provides a thin, fail-safe wrapper around the Langfuse Python SDK.
If Langfuse is disabled or keys are missing, all calls become no-ops
so the pipeline still works without an observability backend.

Usage in pipeline.py:
    from src.tracing import create_trace
    trace = create_trace(name="answer", input={"query": q})
    span = trace.span(name="dense_retrieval", ...)
    span.end(output={...})
    trace.update(output={...})
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from time import perf_counter

from src.config import (
    LANGFUSE_ENABLED,
    LANGFUSE_HOST,
    LANGFUSE_PUBLIC_KEY,
    LANGFUSE_SECRET_KEY,
)

logger = logging.getLogger(__name__)

# ── Singleton Langfuse client ────────────────────────────────────────────
_langfuse_client = None


def _get_langfuse():
    """Lazily initialise the Langfuse client (singleton)."""
    global _langfuse_client
    if _langfuse_client is not None:
        return _langfuse_client

    if not LANGFUSE_ENABLED:
        logger.info("Langfuse tracing disabled via LANGFUSE_ENABLED=false")
        return None

    if not LANGFUSE_PUBLIC_KEY or not LANGFUSE_SECRET_KEY:
        logger.warning(
            "Langfuse keys not set (LANGFUSE_PUBLIC_KEY / LANGFUSE_SECRET_KEY). "
            "Tracing will be disabled."
        )
        return None

    try:
        from langfuse import Langfuse

        _langfuse_client = Langfuse(
            public_key=LANGFUSE_PUBLIC_KEY,
            secret_key=LANGFUSE_SECRET_KEY,
            host=LANGFUSE_HOST,
            # Async flush — does not block the request path
            flush_at=5,
            flush_interval=5,
        )
        logger.info("Langfuse client initialised (host=%s)", LANGFUSE_HOST)
        return _langfuse_client
    except Exception as exc:
        logger.warning("Failed to initialise Langfuse: %s — tracing disabled", exc)
        return None


def flush():
    """Flush any buffered traces. Call on shutdown or after test runs."""
    client = _get_langfuse()
    if client is not None:
        try:
            client.flush()
        except Exception:
            pass


# ── Trace / Span helpers ────────────────────────────────────────────────

class _NoOpObj:
    """Quacks like a Langfuse trace/span but does nothing."""

    def span(self, **kwargs):
        return _NoOpObj()

    def generation(self, **kwargs):
        return _NoOpObj()

    def update(self, **kwargs):
        return self

    def end(self, **kwargs):
        return self

    def score(self, **kwargs):
        return self

    @property
    def trace_id(self):
        return "noop"


_NOOP = _NoOpObj()


def create_trace(*, name: str, input: dict | None = None, metadata: dict | None = None, user_id: str | None = None):
    """Create a new Langfuse trace. Returns a no-op object if Langfuse is off."""
    client = _get_langfuse()
    if client is None:
        return _NOOP
    try:
        return client.trace(
            name=name,
            input=input,
            metadata=metadata or {},
            user_id=user_id,
        )
    except Exception as exc:
        logger.debug("Langfuse trace creation failed: %s", exc)
        return _NOOP


@contextmanager
def timed_span(parent, *, name: str, input: dict | None = None, metadata: dict | None = None):
    """Context manager that creates a span, times it, and yields a dict to populate output.

    Example:
        with timed_span(trace, name="rerank", input={"n": 20}) as ctx:
            results = do_rerank(...)
            ctx["output"] = {"kept": len(results)}
    """
    output_holder: dict = {"output": None, "metadata": metadata or {}}
    start = perf_counter()
    try:
        span = parent.span(name=name, input=input, metadata=metadata or {})
    except Exception:
        span = _NOOP
    try:
        yield output_holder
    finally:
        elapsed_ms = int((perf_counter() - start) * 1000)
        try:
            out = output_holder.get("output") or {}
            out["latency_ms"] = elapsed_ms
            span.update(output=out)
            span.end()
        except Exception:
            pass
