"""
Evaluation script for DocuRAG (Milestone 9).

Runs queries from golden_qa.jsonl through the pipeline and computes metrics.
Supports a `--limit` flag for quick testing.
"""

import json
import argparse
import sys
import os
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from src.pipeline import answer

def load_golden_qa(limit: int = None):
    dataset_path = Path(__file__).parent / "golden_qa.jsonl"
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            data.append(json.loads(line))
            if limit and len(data) >= limit:
                break
    return data


def run_eval(limit: int = None, category_filter: str = None):
    data = load_golden_qa(limit)
    if category_filter:
        data = [d for d in data if d["category"] == category_filter or d["category"] == "Unsupported"]
    
    print(f"Loaded {len(data)} questions for evaluation (Filter: {category_filter}).")
    
    metrics = {
        "total": len(data),
        "supported_total": 0,
        "unsupported_total": 0,
        "answer_match_count": 0,    # Answer contains expected snippet
        "retrieval_hit_5_count": 0, # Any citation/retrieved chunk contains expected snippet
        "correct_refusal_count": 0  # Unsupported query correctly refused
    }
    
    for item in data:
        query = item["query"]
        expected = item["expected_answer_snippet"]
        category = item["category"]
        
        print(f"\nEvaluating: [{category}] {query}")
        
        # We use use_cache=True for eval to speed up reruns, though for pristine eval it can be False
        response = answer(query, use_cache=True)
        ans = response.answer.lower()
        exp = expected.lower()
        
        hit_5 = False
        
        if category == "Unsupported":
            metrics["unsupported_total"] += 1
            if "don't have enough information" in ans:
                metrics["correct_refusal_count"] += 1
                print("  -> CORRECT REFUSAL")
            else:
                print(f"  -> FAILED REFUSAL (Answered: {ans[:50]}...)")
        else:
            metrics["supported_total"] += 1
            
            # Answer Match
            if exp in ans:
                metrics["answer_match_count"] += 1
                print(f"  -> ANSWER MATCH (Found: {expected})")
            else:
                print(f"  -> ANSWER MISS (Expected: {expected}, Got: {ans[:50]}...)")
                
            # Retrieval Hit@5 Approximation: Did we cite anything? (Real Ragas would use embeddings)
            if len(response.citations) > 0:
                 metrics["retrieval_hit_5_count"] += 1
                 hit_5 = True
            
            print(f"  -> RETRIEVAL HIT@5: {hit_5}")

    print("\n" + "="*40)
    print("EVALUATION RESULTS")
    print("="*40)
    
    supp = metrics["supported_total"]
    unsupp = metrics["unsupported_total"]
    
    if supp > 0:
        ans_acc = (metrics["answer_match_count"] / supp) * 100
        hit5_acc = (metrics["retrieval_hit_5_count"] / supp) * 100
        print(f"Supported Answer Accuracy:   {ans_acc:.1f}% ({metrics['answer_match_count']}/{supp})")
        print(f"Approximate Retrieval Hit@5: {hit5_acc:.1f}% ({metrics['retrieval_hit_5_count']}/{supp})")
        
    if unsupp > 0:
        ref_acc = (metrics["correct_refusal_count"] / unsupp) * 100
        print(f"Refusal Accuracy (Gating):   {ref_acc:.1f}% ({metrics['correct_refusal_count']}/{unsupp})")

    # ── Enforce Metrics Thresholds ─────────────────────────────────────
    print("\n" + "="*40)
    print("OBSERVABILITY METRICS CHECK")
    print("="*40)
    
    from src.observability.metrics import get_citation_coverage, get_failure_rate, _close_conn
    # Query the last 1 hour of local DB (which contains our eval run)
    cov = get_citation_coverage(hours=1.0)
    fail = get_failure_rate(hours=1.0)
    
    print(f"Citation Coverage: {cov['coverage_pct']:.1f}% (Threshold: > 70%)")
    print(f"Failure Rate:      {fail['failure_pct']:.1f}% (Threshold: < 20%)")
    
    _close_conn()
    
    passed = True
    if cov['coverage_pct'] < 70.0:
        print("[FAIL] Citation coverage too low!")
        passed = False
    if fail['failure_pct'] > 20.0:
        print("[FAIL] Failure rate too high!")
        passed = False
    if supp > 0 and ans_acc < 60.0:
        print("[FAIL] Answer Accuracy (Faithfulness proxy) too low!")
        passed = False
        
    if not passed:
        print("\n[FAIL] EVALUATION FAILED.")
        sys.exit(1)
    
    print("\n[PASS] EVALUATION PASSED.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=None, help="Limit number of eval questions")
    parser.add_argument("--category", type=str, default=None, help="Filter by category")
    args = parser.parse_args()
    
    run_eval(args.limit, args.category)
