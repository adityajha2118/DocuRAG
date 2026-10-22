"""
Tests for observability metrics aggregation (Milestone 14).
Verifies percentile math, cost estimation, and failure rate tracking
using a synthetic in-memory SQLite database.
"""

import os
import tempfile
from datetime import datetime, timedelta, timezone
from src.observability import metrics

def test_cost_estimation():
    # Test free tier
    assert metrics.estimate_cost("deepseek-ai/deepseek-v4.1-flash", 1000, 1000) == 0.0
    # Test paid tier
    assert metrics.estimate_cost("gpt-4o-mini", 1000, 1000) == 0.75  # 0.15 + 0.60
    # Test fallback
    assert metrics.estimate_cost("unknown-model", 5000, 5000) == 0.0


def test_percentile_math():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name
    try:
        # Insert 10 requests with latencies 100, 200... 1000
        for i in range(1, 11):
            metrics.record_request(
                query=f"q{i}",
                total_latency_ms=i * 100,
                stage_latencies={"dense_retrieval": i * 10},
                db_path=db_path
            )

        # P50 of [100,200,300,400,500,600,700,800,900,1000] is 500 (index 5)
        # P95 is 900 (index 9)
        res = metrics.get_latency_percentiles(percentiles=[50, 95], db_path=db_path)
        assert res["count"] == 10
        assert res["p50"] == 600
        assert res["p95"] == 1000

        # Stage latencies
        s_res = metrics.get_stage_latency_percentiles("dense_retrieval", percentiles=[50, 95], db_path=db_path)
        assert s_res["count"] == 10
        assert s_res["p50"] == 60
        assert s_res["p95"] == 100
    finally:
        metrics._close_conn(db_path)
        try:
            os.remove(db_path)
        except OSError:
            pass


def test_aggregations():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name
    try:
        # 1. Successful request with valid citations
        metrics.record_request(
            query="q1", total_latency_ms=1000, stage_latencies={},
            n_citations=2, citations_valid=True, refused=False, errored=False, db_path=db_path
        )
        # 2. Refused request
        metrics.record_request(
            query="q2", total_latency_ms=500, stage_latencies={},
            n_citations=0, citations_valid=True, refused=True, errored=False, db_path=db_path
        )
        # 3. Invalid citation request
        metrics.record_request(
            query="q3", total_latency_ms=1500, stage_latencies={},
            n_citations=3, citations_valid=False, refused=False, errored=False, db_path=db_path
        )
        # 4. Errored request (citations_valid=True so it doesn't double count as invalid)
        metrics.record_request(
            query="q4", total_latency_ms=200, stage_latencies={},
            n_citations=0, citations_valid=True, refused=False, errored=True, db_path=db_path
        )

        cov = metrics.get_citation_coverage(db_path=db_path)
        # non-refused non-errored = q1, q3. Valid = q1. Coverage = 50%
        assert cov["total_answerable"] == 2
        assert cov["valid_count"] == 1
        assert cov["coverage_pct"] == 50.0

        fail = metrics.get_failure_rate(db_path=db_path)
        # total=4, errors=1, invalid=1 => failures=2. Failure rate = 50%
        assert fail["total"] == 4
        assert fail["error_count"] == 1
        assert fail["invalid_citation_count"] == 1
        assert fail["failure_pct"] == 50.0
    finally:
        metrics._close_conn(db_path)
        try:
            os.remove(db_path)
        except OSError:
            pass


def test_time_window():
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tf:
        db_path = tf.name
    try:
        conn = metrics._get_conn(db_path)
        old_time = (datetime.now(timezone.utc) - timedelta(hours=5)).isoformat()
        new_time = datetime.now(timezone.utc).isoformat()
        
        # Insert old request
        conn.execute(
            "INSERT INTO request_metrics (timestamp, query, total_latency_ms, stage_latencies, model) VALUES (?, ?, ?, ?, ?)",
            (old_time, "old", 1000, "{}", "test")
        )
        # Insert new request
        conn.execute(
            "INSERT INTO request_metrics (timestamp, query, total_latency_ms, stage_latencies, model) VALUES (?, ?, ?, ?, ?)",
            (new_time, "new", 500, "{}", "test")
        )
        conn.commit()

        # All time
        res_all = metrics.get_latency_percentiles(db_path=db_path)
        assert res_all["count"] == 2

        # Last 1 hour
        res_1h = metrics.get_latency_percentiles(hours=1.0, db_path=db_path)
        assert res_1h["count"] == 1
        assert res_1h["p50"] == 500
    finally:
        metrics._close_conn(db_path)
        try:
            os.remove(db_path)
        except OSError:
            pass
