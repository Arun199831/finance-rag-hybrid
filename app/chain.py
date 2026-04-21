import os
import time
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_core.documents import Document


# ── Prompt ─────────────────────────────────────────────────────

RAG_PROMPT = ChatPromptTemplate.from_template("""
You are a financial analyst assistant. Answer the question using ONLY
the context provided below. If the answer is not in the context,
say "I don't have enough information to answer this question."
Do not use any prior knowledge outside the provided context.

Context:
{context}

Question:
{question}

Answer:
""")


# ── LLM ────────────────────────────────────────────────────────

def get_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0,
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )


# ── Context formatter ──────────────────────────────────────────

def format_docs(docs: list[Document]) -> str:
    return "\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )


# ── Chain builder ──────────────────────────────────────────────

def build_rag_chain(retriever):
    llm = get_llm()

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )
    return chain


# ── Query runner ───────────────────────────────────────────────

async def run_query(
    question: str,
    retriever,
) -> dict:
    start = time.time()

    llm = get_llm()

    # retrieve docs first so we can return them as sources
    docs = await retriever.ainvoke(question)

    chain = (
        {
            "context": lambda _: format_docs(docs),
            "question": RunnablePassthrough()
        }
        | RAG_PROMPT
        | llm
        | StrOutputParser()
    )

    answer = await chain.ainvoke(question)
    latency_ms = (time.time() - start) * 1000

    return {
        "answer": answer,
        "docs": docs,
        "latency_ms": round(latency_ms, 2)
    }