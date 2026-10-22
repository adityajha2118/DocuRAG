"""
Milestone 13 verification — Run a handful of queries through the
instrumented pipeline and export trace info.

Usage:
    python scripts/test_tracing.py
"""

import sys
import os
import json
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dotenv import load_dotenv
load_dotenv()

from src.pipeline import answer
from src.tracing import flush as flush_traces, _get_langfuse

TEST_QUERIES = [
    "How do I declare a path parameter with a type in FastAPI?",
    "How can I return a custom status code in FastAPI?",
    "What is the capital of France?",  # unsupported — should be refused
]


def main():
    print("=" * 60)
    print("  Milestone 13 — Tracing Verification")
    print("=" * 60)

    lf = _get_langfuse()
    if lf:
        print(f"\n  Langfuse ENABLED (host: {os.getenv('LANGFUSE_HOST', 'N/A')})")
    else:
        print("\n  Langfuse NOT connected (keys missing or disabled).")
        print("  Pipeline will run normally; traces go to no-op.")

    results = []
    for i, q in enumerate(TEST_QUERIES, 1):
        print(f"\n[{i}/{len(TEST_QUERIES)}] {q}")
        start = time.perf_counter()
        resp = answer(q, use_cache=False)
        elapsed = time.perf_counter() - start

        row = {
            "query": q,
            "answer_snippet": resp.answer[:120],
            "n_citations": len(resp.citations),
            "n_retrieved": len(resp.retrieved_chunks),
            "refused": "don't have enough information" in resp.answer.lower(),
            "latency_s": round(elapsed, 2),
        }
        results.append(row)
        print(f"    Answer: {row['answer_snippet']}...")
        print(f"    Citations: {row['n_citations']} | Refused: {row['refused']} | Latency: {row['latency_s']}s")

    # Flush any buffered Langfuse traces
    flush_traces()

    # Save summary
    out_path = os.path.join(os.path.dirname(__file__), "..", "eval", "tracing_test_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    print("\n" + "=" * 60)
    print("  TRACING VERIFICATION COMPLETE")
    if lf:
        print("  Check Langfuse UI at http://localhost:3000 for traces!")
    else:
        print("  Start Langfuse with: docker compose up langfuse langfuse-db -d")
        print("  Then add keys to .env and re-run this script.")
    print("=" * 60)


if __name__ == "__main__":
    main()
