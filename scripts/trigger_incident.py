import os
import sys
from pathlib import Path

# Force the regression flag ON before importing config
os.environ["DEBUG_REGRESSION_BAD_PROMPT"] = "True"
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import answer
from src.tracing import flush

queries = [
    "How do I install Langfuse locally?",
    "What is the maximum token limit for the reranker?",
    "How does DocuRAG handle evaluation metrics?",
    "What models are supported for embeddings?",
    "How is the BM25 index built?",
]

def main():
    print("Triggering incident: Running queries with degraded prompt")
    for q in queries:
        print(f"\nQuery: {q}")
        response = answer(q)
        print(f"Answer: {response.answer}")
        print(f"Citations Valid: {len(response.citations) > 0}")
    
    # Flush Langfuse traces
    flush()
    print("\nIncident queries completed and traces flushed.")

if __name__ == "__main__":
    main()
