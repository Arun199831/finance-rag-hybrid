from pydantic import BaseModel, Field
from typing import Optional


# ── Request schemas ────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=5, description="The question to ask the RAG pipeline")
    top_k: int = Field(default=4, ge=1, le=10, description="Number of documents to retrieve")
    retriever_mode: str = Field(default="hybrid", description="hybrid | dense | bm25")


class IngestRequest(BaseModel):
    source: str = Field(..., description="URL or local file path to ingest")
    source_type: str = Field(default="url", description="url | pdf")


# ── Response schemas ───────────────────────────────────────────

class SourceDocument(BaseModel):
    content: str = Field(..., description="The retrieved chunk content")
    source: str = Field(..., description="Where this chunk came from")
    relevance_score: float = Field(..., description="RRF or similarity score")


class QueryResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceDocument]
    retriever_mode: str
    latency_ms: float


class IngestResponse(BaseModel):
    status: str
    documents_added: int
    source: str


class HealthResponse(BaseModel):
    status: str                          
    vectorstore_ready: bool
    llm_ready: bool


class EvalMetrics(BaseModel):
    context_precision: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    retriever_mode: str


class EvalResponse(BaseModel):
    status: str
    metrics: EvalMetrics
    sample_size: int