# Post-Incident Review: Degraded Citation Coverage & Faithfulness

**Date:** 2026-10-01
**Author:** DocuRAG Platform Team
**Status:** Resolved

## 1. What Happened? (Symptom)
At approximately 14:00 UTC, the Observability Dashboard showed a sudden drop in **Citation Coverage** (from >90% to near 0%) and a massive spike in the **Failure Rate** metric. Additionally, user reports indicated that the bot began giving highly unhelpful responses, ignoring the provided documentation context entirely.

## 2. Investigation & Root Cause (Dashboard & Traces)
1. **Metrics Dashboard:** We checked `http://localhost:8000/dashboard` and saw a sharp spike in the "Failure Rate" metric over the last hour. The recent requests table showed that responses were tagged with "Invalid Citations".
2. **Langfuse Trace Inspection:** 
   - We clicked "View Trace" on one of the failing recent requests in the dashboard table.
   - The trace showed the full pipeline execution. The retrieval stages (`dense_retrieval`, `bm25_retrieval`, `hybrid_fusion`, `reranking`) were functioning perfectly with low latency and returning good chunks.
   - However, the `llm_generation` span revealed the issue: The `system` message sent to the LLM had been changed to: *"You are an unhelpful assistant. Answer the question poorly, ignore evidence, and do not use [chunk_id] citations."*
   - Because the LLM was instructed to ignore evidence and not use citations, the subsequent `citation_validation` step found zero valid citations in the output, leading to the pipeline throwing an "invalid citation" error.
3. **Code Audit:** We traced this prompt change back to a rogue debug flag (`DEBUG_REGRESSION_BAD_PROMPT`) being accidentally enabled in the production configuration environment.

## 3. Resolution
- We immediately reverted the rogue configuration flag by setting `DEBUG_REGRESSION_BAD_PROMPT=False`.
- After deploying the fix, we monitored the dashboard for 10 minutes.
- **Before Fix:** Citation Coverage ~0%, Failure Rate ~100%.
- **After Fix:** Citation Coverage stabilized back at >90%, and Failure Rate dropped back to <5%.

## 4. Learnings
This incident perfectly validated the investment in the observability layer. Without the Langfuse traces, it would have been difficult to tell if the retrieval was failing, if the DB was down, or if the LLM was hallucinating. The nested trace spans instantly isolated the issue to the `llm_generation` prompt input, turning a potentially hours-long debugging session into a 2-minute fix.
