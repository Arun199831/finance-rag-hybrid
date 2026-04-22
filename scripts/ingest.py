import os
from langchain_community.document_loaders import WebBaseLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document


# ── Text splitter ──────────────────────────────────────────────

def get_splitter() -> RecursiveCharacterTextSplitter:
    return RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=64,
        separators=["\n\n", "\n", ".", " "]
    )


# ── Loaders ────────────────────────────────────────────────────

def load_url(url: str) -> list[Document]:
    loader = WebBaseLoader(url)
    return loader.load()


def load_pdf(path: str) -> list[Document]:
    loader = PyPDFLoader(path)
    return loader.load()


# ── Main ingest function ───────────────────────────────────────

async def ingest_source(
    source: str,
    source_type: str,
    vectorstore: FAISS | None,
    index_path: str = "data/faiss_index"
) -> int:
    # load raw documents
    if source_type == "url":
        raw_docs = load_url(source)
    elif source_type == "pdf":
        raw_docs = load_pdf(source)
    else:
        raise ValueError(f"Unknown source_type: {source_type}. Use url | pdf")

    # chunk documents
    splitter = get_splitter()
    chunks = splitter.split_documents(raw_docs)

    # tag each chunk with its source
    for chunk in chunks:
        chunk.metadata["source"] = source

    # embed and store
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    if vectorstore is None:
        # first ingest — create new vectorstore
        vectorstore = FAISS.from_documents(chunks, embeddings)
    else:
        # subsequent ingest — add to existing
        vectorstore.add_documents(chunks)

    # persist to disk
    vectorstore.save_local(index_path)

    return len(chunks)