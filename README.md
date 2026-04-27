# Finance RAG — Hybrid Retrieval API

> Production RAG pipeline with BM25 + Dense hybrid retrieval, served via FastAPI, containerised with Docker, and evaluated with RAGAs.

---

## What this is

A finance-domain question answering system built to production standards — not a tutorial demo.

It retrieves relevant documents using hybrid retrieval (BM25 + Dense vectors fused with Reciprocal Rank Fusion), generates grounded answers via GPT-4o-mini, and exposes measurable quality metrics through a live eval endpoint.

Built to demonstrate end-to-end AI engineering — from document ingestion to containerised deployment with continuous quality measurement.

---

## Architecture

## Architecture

'''
                    ┌─────────────────────────────────────────┐
                    │           Docker Container               │
                    │                                          │
Client Request ────►│  FastAPI                                 │
                    │    │                                     │
                    │    ├──► BM25 Retriever (keyword)         │
                    │    │         │                           │
                    │    ├──► Dense Retriever (FAISS)    ──────┼──► FAISS Index (volume)
                    │    │         │                           │
                    │    └──► RRF Fusion                       │
                    │              │                           │
                    │              ▼                           │
                    │         Top-K Docs                       │
                    │              │                           │
                    │              ▼                           │
                    │    LLM (GPT-4o-mini) ◄── Prompt          │
                    │              │                           │
                    │              ▼                           │
                    │    Answer + Sources + Latency            │
                    │                                          │
                    │    RAGAs Eval (/eval endpoint)           │
                    └─────────────────────────────────────────┘

Everything runs inside a single Docker container. FAISS index persists via volume mount.

---

## RAGAs Evaluation Scores

Scores measured on 5-question finance eval dataset. Eval is exposed as a live API endpoint — not a one-off script.

## RAGAs Evaluation Scores

Scores measured on 5-question finance eval dataset with exact-identifier queries (FOMC, NVDA, 13F).

| Metric | Dense only | Hybrid (BM25 + Dense) |
|---|---|---|
| Context Precision | 0.7500 | 0.7611 |
| Context Recall | 0.6000 | 0.6000 |
| Faithfulness | 0.9778 | 0.7778 |
| Answer Relevancy | 0.6571 | 0.6898 |

> Hybrid retrieval improves precision and answer relevancy on exact-keyword finance queries.
> Full differentiation expected on SEC filings and earnings reports corpus.



---
## Agentic RAG — LangGraph

The `/agent` endpoint uses a LangGraph state graph instead of a plain chain.

retrieve → check_quality → generate → validate_answer → END
↓
end_empty → END (if no good docs found)

Each node has one job. The graph decides the path based on what it finds — if retrieval returns poor content, it routes to a fallback instead of wasting an LLM call on empty context. After generation, a validation node checks the answer for hallucination signals.

## Stack

| Layer | Technology |
|---|---|
| Retrieval | BM25 (rank-bm25) + FAISS, fused with RRF |
| Orchestration | LangChain, LangChain Community |
| LLM | GPT-4o-mini via OpenAI |
| Serving | FastAPI, Uvicorn, Pydantic |
| Evaluation | RAGAs (4 metrics) |
| Infrastructure | Docker, docker-compose |
| Package management | uv |

---

## Quick Start

**Requirements:** Docker Desktop, OpenAI API key

```bash
git clone https://github.com/Arun199831/finance-rag-hybrid.git
cd finance-rag-hybrid

# Add your OpenAI API key
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=your-key-here

# Start the API
docker compose up --build
```

API is live at `http://localhost:8000`
Interactive docs at `http://localhost:8000/docs`

---

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/health` | GET | Liveness + readiness check |
| `/ingest` | POST | Add a URL or PDF to the vector store |
| `/query` | POST | Ask a question, get answer + sources |
| `/eval` | GET | Run RAGAs evaluation, returns live scores |
| `/agent` | POST | LangGraph agentic RAG with quality checks |

---

## Usage Examples

### Ingest a document
```bash
curl -X POST http://localhost:8000/ingest \
  -H "Content-Type: application/json" \
  -d '{"source": "https://en.wikipedia.org/wiki/Federal_Reserve", "source_type": "url"}'
```

### Query the pipeline
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What does the Federal Reserve do?",
    "retriever_mode": "hybrid",
    "top_k": 4
  }'
```

### Check health
```bash
curl http://localhost:8000/health
```

### Run evaluation
```bash
curl http://localhost:8000/eval
```

---

## Example Response

```json
{
  "question": "What does the Federal Reserve do?",
  "answer": "The Federal Reserve serves as the central bank of the United States...",
  "sources": [
    {
      "content": "The Federal Reserve System is the central banking system...",
      "source": "https://en.wikipedia.org/wiki/Federal_Reserve",
      "relevance_score": 0.87
    }
  ],
  "retriever_mode": "hybrid",
  "latency_ms": 1842.3
}
```

---

## Why Hybrid Retrieval

Pure dense retrieval struggles on exact identifiers — ticker symbols like `NVDA`, filing codes like `13F`, rate figures like `5.25%`. Embeddings generalise meaning, which blurs exact terms into semantic neighbours.

BM25 catches exact keyword matches. Dense retrieval catches semantic meaning. RRF fusion promotes documents that rank well in both — rewarding consensus over individual dominance.

A document ranking 3rd in both retrievers beats one ranking 1st in only one.

---

## Project Structure

finance-rag-hybrid/
├── app/
│   ├── main.py          # FastAPI app, lifespan, routes
│   ├── retriever.py     # BM25 + Dense + RRF logic
│   ├── chain.py         # LangChain RAG chain
│   ├── schemas.py       # Pydantic request/response models
│   └── eval.py          # RAGAs evaluation logic
├── data/
│   ├── faiss_index/     # persisted via Docker volume
│   └── eval_dataset.json
├── scripts/
│   └── ingest.py        # document ingestion
├── tests/
│   └── test_retriever.py
├── Dockerfile
├── docker-compose.yml
└── README.md

---

## Background

Built as part of a structured RAG learning path.
Related: [rag-mastery](https://github.com/Arun199831/rag-mastery)

