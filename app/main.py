import os
from contextlib import asynccontextmanager
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from app.schemas import (
    QueryRequest, QueryResponse, SourceDocument,
    IngestRequest, IngestResponse,
    HealthResponse, EvalResponse, EvalMetrics
)
from app.retriever import load_vectorstore, get_retriever
from app.chain import run_query
from app.eval import run_eval

load_dotenv()


# ── Lifespan ───────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Starting up — loading vectorstore...")
    try:
        app.state.vectorstore = load_vectorstore()
        app.state.docs = list(app.state.vectorstore.docstore._dict.values())
        app.state.vectorstore_ready = True
        print(f"Vectorstore loaded — {len(app.state.docs)} documents")
    except Exception as e:
        print(f"Vectorstore load failed: {e}")
        app.state.vectorstore_ready = False
        app.state.vectorstore = None
        app.state.docs = []

    yield

    print("Shutting down...")


# ── App ────────────────────────────────────────────────────────

app = FastAPI(
    title="Finance RAG — Hybrid Retrieval API",
    description="Production RAG pipeline with BM25 + Dense hybrid retrieval and RAGAs evaluation",
    version="1.0.0",
    lifespan=lifespan
)


# ── Routes ─────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok" if app.state.vectorstore_ready else "degraded",
        vectorstore_ready=app.state.vectorstore_ready,
        llm_ready=bool(os.getenv("OPENAI_API_KEY"))
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    if not app.state.vectorstore_ready:
        raise HTTPException(
            status_code=503,
            detail="Vectorstore not ready. Run /ingest first."
        )
    try:
        retriever = get_retriever(
            mode=request.retriever_mode,
            vectorstore=app.state.vectorstore,
            docs=app.state.docs,
            top_k=request.top_k
        )
        result = await run_query(request.question, retriever)
        sources = [
            SourceDocument(
                content=doc.page_content,
                source=doc.metadata.get("source", "unknown"),
                relevance_score=doc.metadata.get("relevance_score", 0.0)
            )
            for doc in result["docs"]
        ]
        return QueryResponse(
            question=request.question,
            answer=result["answer"],
            sources=sources,
            retriever_mode=request.retriever_mode,
            latency_ms=result["latency_ms"]
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")


@app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    try:
        from scripts.ingest import ingest_source
        count = await ingest_source(
            source=request.source,
            source_type=request.source_type,
            vectorstore=app.state.vectorstore
        )

        # reload vectorstore from disk after ingest
        app.state.vectorstore = load_vectorstore()
        app.state.docs = list(app.state.vectorstore.docstore._dict.values())
        app.state.vectorstore_ready = True

        return IngestResponse(
            status="success",
            documents_added=count,
            source=request.source
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ingest error: {str(e)}")


@app.get("/eval", response_model=EvalResponse)
async def eval():
    if not app.state.vectorstore_ready:
        raise HTTPException(
            status_code=503,
            detail="Vectorstore not ready. Run /ingest first."
        )
    try:
        retriever = get_retriever(
    mode="dense",
    vectorstore=app.state.vectorstore,
    docs=app.state.docs
)

        metrics = await run_eval(retriever, mode="dense")
        return EvalResponse(
            status="success",
            metrics=EvalMetrics(**metrics),
            sample_size=metrics.get("sample_size", 0)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Eval error: {str(e)}")