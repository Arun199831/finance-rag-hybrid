import os
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# ── Embedding model ────────────────────────────────────────────

def get_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


# ── Load vector store ──────────────────────────────────────────

def load_vectorstore(index_path: str = "data/faiss_index") -> FAISS:
    embeddings = get_embeddings()
    return FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )


# ── Build retrievers ───────────────────────────────────────────

def build_dense_retriever(vectorstore: FAISS, top_k: int = 4):
    return vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": top_k}
    )


def build_bm25_retriever(docs: list[Document], top_k: int = 4) -> BM25Retriever:
    retriever = BM25Retriever.from_documents(docs)
    retriever.k = top_k
    return retriever


def build_hybrid_retriever(
    vectorstore: FAISS,
    docs: list[Document],
    top_k: int = 4,
    bm25_weight: float = 0.4,
    dense_weight: float = 0.6
) -> EnsembleRetriever:
    bm25 = build_bm25_retriever(docs, top_k)
    dense = build_dense_retriever(vectorstore, top_k)

    return EnsembleRetriever(
        retrievers=[bm25, dense],
        weights=[bm25_weight, dense_weight]
    )


# ── Retriever factory ──────────────────────────────────────────

def get_retriever(
    mode: str,
    vectorstore: FAISS,
    docs: list[Document],
    top_k: int = 4
):
    if mode == "hybrid":
        return build_hybrid_retriever(vectorstore, docs, top_k)
    elif mode == "dense":
        return build_dense_retriever(vectorstore, top_k)
    elif mode == "bm25":
        return build_bm25_retriever(docs, top_k)
    else:
        raise ValueError(f"Unknown retriever mode: {mode}. Choose hybrid | dense | bm25")