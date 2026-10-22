"""
Offline Evaluation for DocuRAG (Milestone 9).

Computes metrics WITHOUT requiring LLM judge calls:
  1. Semantic Similarity:  Cosine similarity between answer & expected snippet embeddings
  2. Answer Match:         Whether the expected snippet appears in the generated answer
  3. Retrieval Success:    Whether the pipeline retrieved any relevant context
  4. Citation Count:       Average number of citations per answer

Uses the same local sentence-transformer model as the RAG pipeline.
"""

import json
import sys
import os
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from sentence_transformers import SentenceTransformer
from src.pipeline import answer
from dotenv import load_dotenv

load_dotenv()


def cosine_similarity(a, b):
    """Compute cosine similarity between two vectors."""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8))


def run_evaluation():
    print("=" * 60)
    print("  DocuRAG — Offline Evaluation Suite")
    print("=" * 60)

    # Load embedding model (same one used in the RAG pipeline)
    print("\nLoading evaluation embedding model...")
    embed_model = SentenceTransformer("all-MiniLM-L6-v2")

    # Load golden dataset
    with open("eval/golden_qa.jsonl", "r") as f:
        all_data = [json.loads(line) for line in f if line.strip()]
    
    fastapi_data = [d for d in all_data if d["category"] == "FastAPI"]
    unsupported_data = [d for d in all_data if d["category"] == "Unsupported"]

    print(f"\n  FastAPI questions:     {len(fastapi_data)}")
    print(f"  Unsupported questions: {len(unsupported_data)}")
    print(f"  Total:                 {len(fastapi_data) + len(unsupported_data)}\n")

    # ── Evaluate FastAPI questions ──────────────────────────────────────
    results = []
    similarities = []
    answer_matches = 0
    retrieval_hits = 0
    total_citations = 0

    for i, item in enumerate(fastapi_data, 1):
        q = item["query"]
        expected = item["expected_answer_snippet"]
        print(f"[{i}/{len(fastapi_data)}] {q}")

        resp = answer(q)
        ans_text = resp.answer
        
        # 1. Exact substring match
        match = expected.lower() in ans_text.lower()
        if match:
            answer_matches += 1
        
        # 2. Semantic similarity
        emb_answer = embed_model.encode(ans_text)
        emb_expected = embed_model.encode(expected)
        sim = cosine_similarity(emb_answer, emb_expected)
        similarities.append(sim)
        
        # 3. Retrieval success
        n_citations = len(resp.citations) if resp.citations else 0
        total_citations += n_citations
        if n_citations > 0:
            retrieval_hits += 1
        
        status = "[MATCH]" if match else "[MISS]"
        print(f"    {status} | Similarity: {sim:.3f} | Citations: {n_citations}")

        results.append({
            "question": q,
            "expected": expected,
            "answer_snippet": ans_text[:150],
            "exact_match": match,
            "semantic_similarity": round(sim, 4),
            "citation_count": n_citations,
        })

    # ── Evaluate Unsupported questions (Refusal Accuracy) ──────────────
    refusal_correct = 0
    for i, item in enumerate(unsupported_data, 1):
        q = item["query"]
        print(f"[Unsupported {i}/{len(unsupported_data)}] {q}")
        
        resp = answer(q)
        ans_text = resp.answer.lower()
        
        refused = "don't have enough information" in ans_text or "no results" in ans_text or "i cannot" in ans_text
        if refused:
            refusal_correct += 1
            print(f"    [CORRECT REFUSAL]")
        else:
            print(f"    [FAILED REFUSAL] (answered: {ans_text[:60]}...)")

    # ── Aggregate Results ──────────────────────────────────────────────
    n = len(fastapi_data)
    avg_sim = sum(similarities) / n if n > 0 else 0
    ans_acc = (answer_matches / n * 100) if n > 0 else 0
    ret_acc = (retrieval_hits / n * 100) if n > 0 else 0
    avg_cite = total_citations / n if n > 0 else 0
    n_unsupp = len(unsupported_data)
    ref_acc = (refusal_correct / n_unsupp * 100) if n_unsupp > 0 else 0

    print("\n" + "=" * 60)
    print("  EVALUATION RESULTS")
    print("=" * 60)
    print(f"  Questions evaluated:       {n}")
    print(f"  Answer Match Accuracy:     {ans_acc:.1f}%  ({answer_matches}/{n})")
    print(f"  Avg Semantic Similarity:   {avg_sim:.3f}  (0-1 scale)")
    print(f"  Retrieval Hit Rate:        {ret_acc:.1f}%  ({retrieval_hits}/{n})")
    print(f"  Avg Citations per Answer:  {avg_cite:.1f}")
    if n_unsupp > 0:
        print(f"  Refusal Accuracy:          {ref_acc:.1f}%  ({refusal_correct}/{n_unsupp})")
    
    # Overall quality score (weighted combination)
    overall = (ans_acc * 0.3 + avg_sim * 100 * 0.3 + ret_acc * 0.2 + ref_acc * 0.2)
    print(f"")
    print(f"  ** Overall Quality Score:  {overall:.1f}%")
    print("=" * 60)

    # Save detailed results
    out_path = os.path.join(os.path.dirname(__file__), "eval_results.json")
    with open(out_path, "w") as f:
        json.dump({
            "summary": {
                "answer_match_accuracy": round(ans_acc, 1),
                "semantic_similarity": round(avg_sim, 4),
                "retrieval_hit_rate": round(ret_acc, 1),
                "avg_citations": round(avg_cite, 1),
                "refusal_accuracy": round(ref_acc, 1),
                "overall_quality": round(overall, 1),
            },
            "details": results
        }, f, indent=2)
    print(f"\nDetailed results saved to: {out_path}")


if __name__ == "__main__":
    run_evaluation()
