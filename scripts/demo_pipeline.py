"""
DocuRAG — Milestone 5 Pipeline Demonstration.

Runs one successful query and one out-of-corpus query through the real LLM.
"""

import os
import sys

from src.config import LLM_API_KEY

# Ensure API key is available
if not LLM_API_KEY:
    print("ERROR: Please set OPENAI_API_KEY in your .env file or environment.")
    sys.exit(1)

from src.pipeline import answer

def print_response(title: str, resp):
    print(f"\n{'='*80}")
    print(f"[{title}] Query: {resp.query}")
    print(f"{'='*80}")
    print(f"\n{resp.answer}\n")
    if resp.citations:
        print("CITATIONS:")
        for c in resp.citations:
            print(f"  - [{c.chunk_id}] {c.title} ({c.url})")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    print("Running demonstration...")
    
    # 1. Real successful query
    q1 = "How do I declare a path parameter with a type in FastAPI?"
    resp1 = answer(q1, use_cache=False)
    print_response("SUCCESSFUL QUERY", resp1)
    
    # 2. Out-of-corpus / nonsense query
    q2 = "What are the rules of underwater basket weaving in 2024?"
    resp2 = answer(q2, use_cache=False)
    print_response("NONSENSE QUERY", resp2)
