# DocuRAG

DocuRAG is a production-ready, highly accurate Retrieval-Augmented Generation (RAG) system designed to answer technical questions over AI/ML documentation frameworks (PyTorch, Hugging Face, scikit-learn, FastAPI, LangChain, and LangGraph).

## Features

- **Asynchronous Crawling**: Respects `robots.txt`, handles rate limits, and performs smart parsing to extract clean content with headings and code blocks.
- **Semantic Chunking**: Structure-aware chunking preserving paragraph/code block integrity with contextual token overlaps.
- **Hybrid Retrieval**: Combines Dense (ChromaDB) and Sparse (BM25) search using Reciprocal Rank Fusion ($k=60$) for optimal recall.
- **Cross-Encoder Reranking**: Pointwise reranking via `ms-marco-MiniLM-L-6-v2` guarantees the most relevant context is selected.
- **Evidence Gating & Hallucination Prevention**: Strict minimum scoring gates and lexical citation overlap heuristics prevent the model from answering unsupported or out-of-corpus queries.
- **Ragas-Powered Evaluation**: Includes a 50-question Golden Dataset and an automated evaluation script that computes Ragas metrics (Faithfulness, Answer Relevancy, Context Precision) using offline semantic embeddings.
- **Production API**: Fully containerized FastAPI server with structured logging, caching, and health endpoints.
- **CI/CD Integrated**: Configured with GitHub Actions for automated syntax and import checks on every push to main.

## Setup

1. **Install Dependencies**:
   Ensure you have Python 3.11+ installed.
   ```bash
   pip install -r requirements.txt
   ```

2. **Environment Configuration**:
   Create a `.env` file in the root directory:
   ```env
   OPENAI_API_KEY=nvapi-your-nvidia-key-here
   OPENAI_BASE_URL=https://integrate.api.nvidia.com/v1
   LLM_MODEL=deepseek-ai/deepseek-v4.1-flash
   ```

3. **Ingest Documentation** (Currently configured for FastAPI testing):
   ```bash
   python src/ingestion/crawler.py
   python src/ingestion/parser.py
   python src/ingestion/cleaner.py
   python src/ingestion/chunker.py
   python src/retrieval/indexer.py
   ```

4. **Run the API**:
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```
   Or run it via Docker (if Docker Desktop is installed):
   ```bash
   docker-compose up --build
   ```

## Observability Dashboard

DocuRAG includes a lightweight observability layer with a local SQLite metrics database and a built-in dashboard. The dashboard tracks latency (P50/P95), estimated costs, citation coverage, and failure rates over time.

To view the dashboard:
1. Start the API server: `uvicorn src.api:app --host 0.0.0.0 --port 8000`
2. Navigate to [http://localhost:8000/dashboard](http://localhost:8000/dashboard)

**Demo Data:**
To see what the dashboard looks like without running thousands of real queries, you can populate it with synthetic mock data (clearly labeled as synthetic) by running:
```bash
python scripts/generate_synthetic_traffic.py
```
This will inject 7 days of realistic-looking traffic into the local metrics store.

## Architecture Overview

- `src/ingestion/`: Handles web scraping, markdown cleaning, and token-aware semantic chunking.
- `src/retrieval/`: Manages the local ChromaDB dense index, BM25 exact-term search, and cross-encoder reranking.
- `src/generation/`: Wraps LLM interactions using NVIDIA NIM, handles system prompting, and performs strict citation validation heuristics.
- `eval/`: Contains the golden QA dataset and the Ragas-based evaluation pipeline (`ragas_eval.py`).
- `.github/workflows/`: CI/CD pipeline definition for automated testing.

## Evaluation

To test the system's accuracy, run the evaluation pipeline against the golden dataset:
```bash
python eval/ragas_eval.py
```
This runs the pipeline on the test set and calculates semantic similarity, retrieval hit rates, and refusal accuracy to ensure high-fidelity answers.
