"""
DocuRAG — FastAPI Application (Milestone 8).

Exposes a REST API for the RAG pipeline.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from pydantic import BaseModel, Field

from src.pipeline import answer, AnswerResponse
from src.generation.citations import Citation
from src.observability import metrics


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question about the documentation.")
    use_cache: bool = Field(True, description="Whether to use the local LLM response cache.")


class CitationModel(BaseModel):
    chunk_id: str
    title: str
    url: str
    section: str
    source: str


class AnswerResponseModel(BaseModel):
    query: str
    answer: str
    citations: list[CitationModel]


app = FastAPI(
    title="DocuRAG API",
    description="RAG over AI/ML documentation using local embeddings/reranking and cloud LLM generation.",
    version="1.0.0",
)


@app.post("/query", response_model=AnswerResponseModel)
@app.post("/ask", response_model=AnswerResponseModel)
def ask_question(request: QueryRequest):
    """Submit a question to the DocuRAG pipeline."""
    try:
        # Call the pipeline
        resp: AnswerResponse = answer(request.query, use_cache=request.use_cache)
        
        # Convert dataclasses to Pydantic models for serialization
        return AnswerResponseModel(
            query=resp.query,
            answer=resp.answer,
            citations=[
                CitationModel(
                    chunk_id=c.chunk_id,
                    title=c.title,
                    url=c.url,
                    section=c.section,
                    source=c.source,
                )
                for c in resp.citations
            ]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}


@app.get("/")
def read_root():
    """Redirect root to API documentation."""
    return RedirectResponse(url="/docs")


# ── Observability Endpoints ──────────────────────────────────────────────

@app.get("/metrics/summary")
def get_metrics_summary(hours: float = Query(24.0, description="Time window in hours")):
    """Get overall metrics summary for the dashboard."""
    try:
        return metrics.get_summary(hours=hours)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/timeseries")
def get_metrics_timeseries(hours: float = Query(24.0), buckets: int = Query(12)):
    """Get bucketed timeseries metrics for charting."""
    try:
        return {"timeseries": metrics.get_timeseries(hours=hours, buckets=buckets)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/metrics/recent")
def get_recent_requests_api(limit: int = Query(10)):
    """Get the most recent requests for the dashboard table."""
    try:
        return {"requests": metrics.get_recent_requests(limit=limit)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/dashboard", response_class=HTMLResponse)
def get_dashboard():
    """Serve the single-page HTML dashboard."""
    from pathlib import Path
    dashboard_path = Path(__file__).parent / "static" / "dashboard.html"
    if not dashboard_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard UI not found.")
    return dashboard_path.read_text(encoding="utf-8")


@app.get("/sources")
def list_sources():
    """List all available documentation sources."""
    import json
    from pathlib import Path
    
    metadata_path = Path("data/metadata.jsonl")
    if not metadata_path.exists():
        return {"sources": []}
        
    sources = set()
    with open(metadata_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    data = json.loads(line)
                    if "source" in data:
                        sources.add(data["source"])
                except:
                    pass
    return {"sources": sorted(list(sources))}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=True)
