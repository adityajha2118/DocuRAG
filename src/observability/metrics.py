"""
DocuRAG — Request Metrics Store (Milestone 14).

SQLite-backed per-request metrics recording and aggregation.

Design decision: SQLite over Langfuse stored traces because:
  1. Zero external dependency — works without Docker/Langfuse running
  2. Independently testable with synthetic data
  3. Sub-millisecond writes via WAL mode
  4. Aggregation via SQL is simple and efficient
  5. Satisfies the "lightweight local metrics store" constraint

Schema stores per-stage latency as a JSON column for flexibility
(new pipeline stages can be added without schema migrations).
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path

from src.config import METRICS_DB_PATH, TOKEN_PRICING, LLM_MODEL

logger = logging.getLogger(__name__)

# ── Schema ───────────────────────────────────────────────────────────────
_CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS request_metrics (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp       TEXT    NOT NULL,   -- ISO-8601 UTC
    query           TEXT    NOT NULL,
    total_latency_ms INTEGER NOT NULL,
    stage_latencies  TEXT   NOT NULL,   -- JSON: {"dense_retrieval": 12, "bm25": 5, ...}
    input_tokens     INTEGER DEFAULT 0,
    output_tokens    INTEGER DEFAULT 0,
    estimated_cost_usd REAL DEFAULT 0.0,
    model            TEXT   NOT NULL,
    prompt_version   TEXT   DEFAULT '',
    n_citations      INTEGER DEFAULT 0,
    citations_valid  INTEGER DEFAULT 1, -- 1 = valid citations present, 0 = no valid citations
    refused          INTEGER DEFAULT 0, -- 1 = query was refused by evidence gate
    errored          INTEGER DEFAULT 0, -- 1 = pipeline raised an exception
    error_message    TEXT   DEFAULT ''
);
"""

_CREATE_INDEX = """
CREATE INDEX IF NOT EXISTS idx_metrics_timestamp ON request_metrics(timestamp);
"""


# ── Thread-safe connection pool ──────────────────────────────────────────
_local = threading.local()


def _get_conn(db_path: str | None = None) -> sqlite3.Connection:
    """Get a thread-local SQLite connection (WAL mode for concurrent reads)."""
    path = db_path or METRICS_DB_PATH
    if not hasattr(_local, "conn") or _local.conn is None or _local.db_path != path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(path, check_same_thread=False)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=5000")
        conn.execute(_CREATE_TABLE)
        conn.execute(_CREATE_INDEX)
        conn.commit()
        _local.conn = conn
        _local.db_path = path
    return _local.conn

def _close_conn(db_path: str | None = None) -> None:
    """Close the thread-local connection (useful for testing on Windows)."""
    path = db_path or METRICS_DB_PATH
    if hasattr(_local, "conn") and _local.conn is not None and _local.db_path == path:
        _local.conn.close()
        _local.conn = None
        _local.db_path = None


# ── Cost estimation ──────────────────────────────────────────────────────

def estimate_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    """Estimate cost in USD based on the configurable pricing table.

    Falls back to '_default' pricing if the model is not found.
    """
    pricing = TOKEN_PRICING.get(model, TOKEN_PRICING.get("_default", {"input": 0, "output": 0}))
    cost = (input_tokens / 1000.0) * pricing["input"] + (output_tokens / 1000.0) * pricing["output"]
    return round(cost, 6)


# ── Recording ────────────────────────────────────────────────────────────

def record_request(
    *,
    query: str,
    total_latency_ms: int,
    stage_latencies: dict[str, int],
    input_tokens: int = 0,
    output_tokens: int = 0,
    model: str = "",
    prompt_version: str = "",
    n_citations: int = 0,
    citations_valid: bool = True,
    refused: bool = False,
    errored: bool = False,
    error_message: str = "",
    db_path: str | None = None,
) -> int:
    """Record a single request's metrics to SQLite. Returns the row ID."""
    model = model or LLM_MODEL
    cost = estimate_cost(model, input_tokens, output_tokens)
    now = datetime.now(timezone.utc).isoformat()

    conn = _get_conn(db_path)
    cursor = conn.execute(
        """INSERT INTO request_metrics
           (timestamp, query, total_latency_ms, stage_latencies,
            input_tokens, output_tokens, estimated_cost_usd, model,
            prompt_version, n_citations, citations_valid, refused,
            errored, error_message)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            now, query, total_latency_ms, json.dumps(stage_latencies),
            input_tokens, output_tokens, cost, model,
            prompt_version, n_citations, int(citations_valid), int(refused),
            int(errored), error_message,
        ),
    )
    conn.commit()
    return cursor.lastrowid


# ── Aggregation queries ─────────────────────────────────────────────────

def _time_filter(hours: float | None) -> tuple[str, list]:
    """Build a WHERE clause for a rolling time window."""
    if hours is None:
        return "", []
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
    return "WHERE timestamp >= ?", [cutoff]


def get_latency_percentiles(
    percentiles: list[int] = (50, 95),
    hours: float | None = None,
    db_path: str | None = None,
) -> dict[str, int | None]:
    """Compute overall latency percentiles (P50, P95, etc.) over a time window.

    Returns e.g. {"p50": 123, "p95": 456, "count": 100}.
    """
    where, params = _time_filter(hours)
    conn = _get_conn(db_path)
    rows = conn.execute(
        f"SELECT total_latency_ms FROM request_metrics {where} ORDER BY total_latency_ms",
        params,
    ).fetchall()

    if not rows:
        return {f"p{p}": None for p in percentiles} | {"count": 0}

    values = [r[0] for r in rows]
    n = len(values)
    result = {"count": n}
    for p in percentiles:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        result[f"p{p}"] = values[idx]
    return result


def get_stage_latency_percentiles(
    stage: str,
    percentiles: list[int] = (50, 95),
    hours: float | None = None,
    db_path: str | None = None,
) -> dict[str, int | None]:
    """Compute per-stage latency percentiles over a time window."""
    where, params = _time_filter(hours)
    conn = _get_conn(db_path)
    rows = conn.execute(
        f"SELECT stage_latencies FROM request_metrics {where}", params
    ).fetchall()

    values = []
    for (blob,) in rows:
        stages = json.loads(blob)
        if stage in stages:
            values.append(stages[stage])

    if not values:
        return {f"p{p}": None for p in percentiles} | {"count": 0}

    values.sort()
    n = len(values)
    result = {"count": n}
    for p in percentiles:
        idx = int(n * p / 100)
        idx = min(idx, n - 1)
        result[f"p{p}"] = values[idx]
    return result


def get_average_cost(hours: float | None = None, db_path: str | None = None) -> dict:
    """Average estimated cost per request over a time window."""
    where, params = _time_filter(hours)
    conn = _get_conn(db_path)
    row = conn.execute(
        f"SELECT AVG(estimated_cost_usd), SUM(estimated_cost_usd), COUNT(*) "
        f"FROM request_metrics {where}",
        params,
    ).fetchone()
    return {
        "avg_cost_usd": round(row[0] or 0, 6),
        "total_cost_usd": round(row[1] or 0, 6),
        "count": row[2] or 0,
    }


def get_citation_coverage(hours: float | None = None, db_path: str | None = None) -> dict:
    """Percentage of non-refused requests that had valid citations."""
    where, params = _time_filter(hours)
    conn = _get_conn(db_path)
    # Only count non-refused, non-errored requests
    extra = " AND refused = 0 AND errored = 0"
    if where:
        full_where = where + extra
    else:
        full_where = "WHERE refused = 0 AND errored = 0"

    row = conn.execute(
        f"SELECT COUNT(*), SUM(CASE WHEN citations_valid = 1 AND n_citations > 0 THEN 1 ELSE 0 END) "
        f"FROM request_metrics {full_where}",
        params,
    ).fetchone()
    total = row[0] or 0
    valid = row[1] or 0
    return {
        "coverage_pct": round((valid / total * 100) if total > 0 else 0, 1),
        "valid_count": valid,
        "total_answerable": total,
    }


def get_failure_rate(hours: float | None = None, db_path: str | None = None) -> dict:
    """Failure rate: percentage of requests that errored OR had invalid citations."""
    where, params = _time_filter(hours)
    conn = _get_conn(db_path)
    row = conn.execute(
        f"SELECT COUNT(*), "
        f"SUM(CASE WHEN errored = 1 THEN 1 ELSE 0 END), "
        f"SUM(CASE WHEN citations_valid = 0 AND refused = 0 THEN 1 ELSE 0 END) "
        f"FROM request_metrics {where}",
        params,
    ).fetchone()
    total = row[0] or 0
    errors = row[1] or 0
    invalid = row[2] or 0
    failures = errors + invalid
    return {
        "failure_pct": round((failures / total * 100) if total > 0 else 0, 1),
        "error_count": errors,
        "invalid_citation_count": invalid,
        "total": total,
    }


def get_summary(hours: float | None = None, db_path: str | None = None) -> dict:
    """Full metrics summary suitable for a dashboard endpoint."""
    return {
        "latency": get_latency_percentiles(hours=hours, db_path=db_path),
        "stage_latencies": {
            stage: get_stage_latency_percentiles(stage, hours=hours, db_path=db_path)
            for stage in [
                "dense_retrieval", "bm25_retrieval", "hybrid_fusion",
                "reranking", "evidence_gate", "llm_generation", "citation_validation",
            ]
        },
        "cost": get_average_cost(hours=hours, db_path=db_path),
        "citation_coverage": get_citation_coverage(hours=hours, db_path=db_path),
        "failure_rate": get_failure_rate(hours=hours, db_path=db_path),
    }


def get_timeseries(hours: float = 24.0, buckets: int = 12, db_path: str | None = None) -> list[dict]:
    """Bucket metrics over a time window for charting.
    Returns a list of dicts with aggregated metrics per bucket.
    """
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(hours=hours)
    
    # Calculate bucket edges
    bucket_sec = (hours * 3600) / buckets
    
    # We fetch all rows in the window and bucket them in memory
    # (Since it's SQLite and we expect moderate volume, this is fine for local observability)
    conn = _get_conn(db_path)
    rows = conn.execute(
        """SELECT timestamp, total_latency_ms, estimated_cost_usd, 
                  citations_valid, n_citations, refused, errored 
           FROM request_metrics WHERE timestamp >= ? ORDER BY timestamp""",
        (start_time.isoformat(),)
    ).fetchall()
    
    # Initialize empty buckets
    ts_data = []
    for i in range(buckets):
        b_start = start_time + timedelta(seconds=i * bucket_sec)
        b_end = b_start + timedelta(seconds=bucket_sec)
        ts_data.append({
            "bucket_start": b_start.isoformat(),
            "bucket_end": b_end.isoformat(),
            "latencies": [],
            "cost_sum": 0.0,
            "total_count": 0,
            "answerable_count": 0,
            "valid_citation_count": 0,
            "error_count": 0,
            "invalid_citation_count": 0,
        })
    
    # Populate buckets
    for row in rows:
        ts_str, lat, cost, c_valid, n_cit, refused, errored = row
        dt = datetime.fromisoformat(ts_str)
        # Find bucket
        idx = int((dt - start_time).total_seconds() / bucket_sec)
        if 0 <= idx < buckets:
            b = ts_data[idx]
            b["total_count"] += 1
            b["latencies"].append(lat)
            b["cost_sum"] += cost
            if not refused and not errored:
                b["answerable_count"] += 1
                if c_valid and n_cit > 0:
                    b["valid_citation_count"] += 1
            if errored:
                b["error_count"] += 1
            if not c_valid and not refused:
                b["invalid_citation_count"] += 1
                
    # Finalize aggregates
    result = []
    for b in ts_data:
        latencies = sorted(b["latencies"])
        n = len(latencies)
        p50 = latencies[int(n * 0.5)] if n > 0 else 0
        p95 = latencies[int(n * 0.95)] if n > 0 else 0
        avg_cost = (b["cost_sum"] / n) if n > 0 else 0.0
        
        cov_pct = (b["valid_citation_count"] / b["answerable_count"] * 100) if b["answerable_count"] > 0 else 100.0
        fail_pct = ((b["error_count"] + b["invalid_citation_count"]) / n * 100) if n > 0 else 0.0
        
        result.append({
            "timestamp": b["bucket_start"],
            "p50_latency": p50,
            "p95_latency": p95,
            "avg_cost": round(avg_cost, 6),
            "citation_coverage": round(cov_pct, 1),
            "failure_rate": round(fail_pct, 1),
            "count": n
        })
    return result


def get_recent_requests(limit: int = 10, db_path: str | None = None) -> list[dict]:
    """Fetch the most recent requests for the dashboard table."""
    conn = _get_conn(db_path)
    rows = conn.execute(
        """SELECT id, timestamp, query, total_latency_ms, estimated_cost_usd, 
                  citations_valid, refused, errored 
           FROM request_metrics ORDER BY timestamp DESC LIMIT ?""",
        (limit,)
    ).fetchall()
    
    return [
        {
            "id": r[0],
            "timestamp": r[1],
            "query": r[2],
            "latency_ms": r[3],
            "cost_usd": round(r[4], 4),
            "status": "Error" if r[7] else ("Refused" if r[6] else ("Invalid Citations" if not r[5] else "Success")),
            "trace_id": f"langfuse-{r[0]}" # Mock linking for dashboard
        }
        for r in rows
    ]
