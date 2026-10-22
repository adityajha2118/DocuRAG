import sqlite3
import random
import uuid
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from src.config import METRICS_DB_PATH

def generate_traffic(days: int = 7, requests_per_day: int = 50):
    """Generates synthetic traffic for the dashboard."""
    print(f"Generating synthetic traffic for the last {days} days...")
    
    # Ensure directory exists
    Path(METRICS_DB_PATH).parent.mkdir(parents=True, exist_ok=True)
    
    conn = sqlite3.connect(METRICS_DB_PATH)
    
    # Create table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS request_metrics (
            id TEXT PRIMARY KEY,
            timestamp TEXT,
            query TEXT,
            total_latency_ms INTEGER,
            stage_latencies TEXT,
            input_tokens INTEGER,
            output_tokens INTEGER,
            estimated_cost_usd REAL,
            model TEXT,
            prompt_version TEXT,
            n_citations INTEGER,
            citations_valid INTEGER,
            refused INTEGER,
            errored INTEGER,
            error_message TEXT
        )
    """)
    
    queries = [
        "How do I configure the API?",
        "What is the default reranker model?",
        "How to run evaluation?",
        "Explain hybrid search.",
        "How to deploy with docker?",
        "Can I use Anthropic models?",
        "Nonsense query testing refusal",
        "What is the BM25 weight?",
    ]
    
    now = datetime.now(timezone.utc)
    start_time = now - timedelta(days=days)
    
    inserted = 0
    
    # Standard cost estimate: 1000 in, 200 out for gpt-4o-mini = ~ $0.00027
    
    for day in range(days):
        for _ in range(requests_per_day):
            # Random time within the day
            offset_seconds = random.randint(0, 86400)
            req_time = start_time + timedelta(days=day, seconds=offset_seconds)
            
            query = random.choice(queries)
            
            # Simulate outcomes
            rand = random.random()
            refused = 1 if "Nonsense" in query or rand < 0.05 else 0
            errored = 1 if rand > 0.98 else 0
            citations_valid = 1 if not refused and not errored and rand > 0.1 else 0
            n_citations = random.randint(1, 3) if citations_valid else 0
            
            # Simulate latency
            base_latency = random.randint(800, 1500)
            if refused: base_latency = random.randint(300, 500)
            if errored: base_latency = random.randint(100, 2000)
            
            # Occasional spikes
            if random.random() < 0.02:
                base_latency += random.randint(3000, 5000)
                
            input_tokens = random.randint(500, 1500)
            output_tokens = random.randint(50, 300) if not refused else 20
            estimated_cost = (input_tokens / 1000 * 0.00015) + (output_tokens / 1000 * 0.0006)
            
            stage_latencies = {
                "dense_retrieval": int(base_latency * 0.1),
                "bm25_retrieval": int(base_latency * 0.05),
                "hybrid_fusion": int(base_latency * 0.05),
                "reranking": int(base_latency * 0.3),
                "evidence_gate": int(base_latency * 0.05),
                "llm_generation": int(base_latency * 0.4),
                "citation_validation": int(base_latency * 0.05),
            }
            
            req_id = str(uuid.uuid4())
            
            conn.execute("""
                INSERT INTO request_metrics 
                (id, timestamp, query, total_latency_ms, stage_latencies, input_tokens, output_tokens, estimated_cost_usd, model, prompt_version, n_citations, citations_valid, refused, errored, error_message)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                req_id,
                req_time.isoformat(),
                query,
                base_latency,
                json.dumps(stage_latencies),
                input_tokens,
                output_tokens,
                estimated_cost,
                "gpt-4o-mini-mock",
                "v1",
                n_citations,
                citations_valid,
                refused,
                errored,
                ""
            ))
            inserted += 1
            
    conn.commit()
    conn.close()
    print(f"Successfully generated {inserted} synthetic requests across {days} days.")

if __name__ == "__main__":
    generate_traffic()
