# DocuRAG Observability Guide

Observability is the missing ~70% of a production RAG system. Building a pipeline that can answer a question is straightforward; maintaining its quality over time as data, models, and usage scale requires deep visibility into what happens at every step of generation.

## 1. Why Observability Matters
In traditional deterministic software, failures are binary (it crashed or it didn't). In Generative AI and RAG, failures are often silent:
- An LLM starts hallucinating slightly more frequently due to a prompt tweak.
- The retrieval index is flooded with noisy chunks, causing the LLM to ignore evidence.
- Context window expansions drastically increase the hidden cost per request.

Our observability layer ensures that we can answer three fundamental questions:
1. What happened during this specific query?
2. How is the system performing over time?
3. Did a recent code or prompt change degrade quality?

## 2. Architecture of the Tracing Layer
We implement a dual-layer observability stack:
1. **Langfuse (Deep Tracing):** We self-host Langfuse via Docker Compose. The `src/pipeline.py` wraps every request in a parent trace. Crucially, the pipeline emits nested *spans* for each logical stage (`dense_retrieval`, `bm25_retrieval`, `hybrid_fusion`, `reranking`, `llm_generation`, `citation_validation`). This isolates latency and logic to the exact component.
2. **Local SQLite Metrics (Agile Aggregation):** Because querying distributed tracing platforms for aggregate math can be slow or cost-prohibitive, we dual-write core metadata (latency, cost, token counts, citation success) directly to a local SQLite database (`data/metrics.db`) per request. This powers our fast, local dashboard and CI/CD gating.

## 3. How to Read a Trace in Langfuse
When inspecting a failure (e.g., via the trace ID link in the local dashboard):
1. Open the Langfuse UI (`http://localhost:3000`).
2. Navigate to the trace. You will see a waterfall view.
3. Look at the `dense_retrieval` and `reranking` spans to confirm the correct chunks were retrieved.
4. Look at the `llm_generation` span. Here you will see the exact `system` and `user` prompt passed to the model, and the exact response.
5. We avoid logging the raw full text of every document chunk in the root trace payload to prevent bloat; instead, we log `chunk_id` and semantic scores.

## 4. The Four Tracked Metrics
Our lightweight dashboard (`GET /dashboard`) tracks four vital signs:
- **Latency (P50 / P95):** Measured end-to-end. Tracking P95 specifically helps catch edge cases where large document chunks cause the reranker or LLM to stall.
- **Cost Per Request:** Computed by multiplying the token counts recorded during generation by the static pricing table in `src/config.py`.
- **Citation Coverage:** The percentage of responses that successfully yielded at least one valid citation matching the source documents. This is our primary proxy for "hallucination / groundedness".
- **Failure Rate:** The percentage of requests that either crashed, were refused by the evidence gate, or failed the citation validation step.

## 5. Case Study: The Milestone 16 Incident
To prove the layer works, we conducted a controlled regression drill:
- **The Injection:** We enabled a `DEBUG_REGRESSION_BAD_PROMPT` flag that silently degraded the LLM prompt to ignore evidence.
- **The Symptom:** The local dashboard instantly showed a massive spike in the **Failure Rate** and a drop to ~0% in **Citation Coverage**.
- **The Diagnosis:** Clicking the failing trace in the dashboard brought us to Langfuse. The waterfall clearly showed that `retrieval` and `reranking` were healthy, but the `llm_generation` span contained the sabotaged prompt.
- **The Result:** A potentially hour-long debugging session (trying to guess if the DB was down or embeddings were broken) was solved in two minutes. See [`docs/incident_example.md`](incident_example.md) for the full post-mortem.

## 6. Observability-Driven Regression Gating
The observability layer is tied directly into our CI/CD pipeline:
- **Milestone 11 (Baseline):** We initially built a simple Python eval script to check basic exact-match accuracy against a golden dataset.
- **Milestone 17 (Observability Gating):** We extended the eval script (`eval/run_eval.py`). Now, at the end of the evaluation run, the script queries the local `metrics.db`. If the *Citation Coverage* drops below 70%, or the *Failure Rate* rises above 20%, the CI/CD pipeline explicitly fails (`exit code 1`).
- **Unified Versioning:** Because prompts and configs (`src/config.py`, `prompts/`) are committed in the repository, any pull request that tweaks a prompt goes through this exact same quantitative gate.
- **Full vs Subset:** We maintain a fast, cheap 10-question subset for per-PR gating (`eval.yml`), and a manually triggered 50-question full regression suite (`full_regression.yml`) for comprehensive pre-release checks.

## 7. How to Run the Dashboard Locally
1. Start the API: `uvicorn src.api:app --reload`
2. Generate some realistic history: `python scripts/generate_synthetic_traffic.py`
3. View the dashboard: Open your browser to `http://localhost:8000/dashboard`.
