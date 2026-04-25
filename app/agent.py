from typing import TypedDict, Literal
from functools import partial
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langgraph.graph import StateGraph, END


# ── State ──────────────────────────────────────────────────────

class RAGAgentState(TypedDict):
    question: str
    documents: list[Document]
    answer: str
    retrieval_attempted: bool      # did retrieval run?
    retrieval_successful: bool     # did retrieval find good content?
    hallucination_flagged: bool    # did answer show hallucination signals?


# ── Nodes ──────────────────────────────────────────────────────

def retrieve_node(state: RAGAgentState, retriever) -> dict:
    try:
        docs = retriever.invoke(state["question"])
        return {
            "documents": docs,
            "retrieval_attempted": True
        }
    except Exception as e:
        print(f"Retrieval failed: {e}")
        return {
            "documents": [],
            "retrieval_attempted": False
        }


def check_quality_node(state: RAGAgentState) -> dict:
    docs = state["documents"]

    minimum_length = 50

    for doc in docs:
        content = doc.page_content.strip()

        if len(content) > minimum_length:
            return {"retrieval_successful": True}

    return {"retrieval_successful": False}


def generate_node(state: RAGAgentState, llm) -> dict:
    from app.chain import format_docs, RAG_PROMPT

    context = format_docs(state["documents"])

    if not context.strip():
        return {"answer": "I don't have enough information to answer this question."}

    chain = RAG_PROMPT | llm | StrOutputParser()
    answer = chain.invoke({
        "context": context,
        "question": state["question"]
    })
    return {"answer": answer}


def validate_answer_node(state: RAGAgentState) -> dict:
    answer = state["answer"]

    hallucination_signals = [
        "as of my knowledge",
        "i believe",
        "i think",
        "approximately",
        "i'm not sure but"
    ]

    answer_lower = answer.lower()
    suspected = False                    # ← initialise here

    for signal in hallucination_signals:
        if signal in answer_lower:
            suspected = True
            break

    if suspected:
        return {
            "hallucination_flagged": True,
            "answer": answer + "\n\n Warning: This answer may contain information not grounded in retrieved context."
        }

    return {"hallucination_flagged": False}


def end_empty_node(state: RAGAgentState) -> dict:
    return {
        "answer": "I don't have enough information to answer this question.",
        "hallucination_flagged": False
    }


# ── Conditional edge functions ─────────────────────────────────

def should_generate(state: RAGAgentState) -> Literal["generate", "end_empty"]:
    if state["retrieval_successful"]:
        return "generate"
    else:
        return "end_empty"


# ── Graph builder ──────────────────────────────────────────────

def build_rag_agent(retriever, llm):
    graph = StateGraph(RAGAgentState)

    # nodes
    graph.add_node("retrieve", partial(retrieve_node, retriever=retriever))
    graph.add_node("check_quality", check_quality_node)
    graph.add_node("generate", partial(generate_node, llm=llm))
    graph.add_node("validate_answer", validate_answer_node)
    graph.add_node("end_empty", end_empty_node)

    # edges
    graph.add_edge("retrieve", "check_quality")
    graph.add_conditional_edges(
        "check_quality",
        should_generate,
        {
            "generate": "generate",
            "end_empty": "end_empty"
        }
    )
    graph.add_edge("generate", "validate_answer")
    graph.add_edge("validate_answer", END)
    graph.add_edge("end_empty", END)

    # entry point
    graph.set_entry_point("retrieve")

    return graph.compile()


# ── Run agent ──────────────────────────────────────────────────

async def run_agent(question: str, retriever, llm) -> dict:
    agent = build_rag_agent(retriever, llm)

    initial_state = RAGAgentState(
        question=question,
        documents=[],
        answer="",
        retrieval_attempted=False,
        retrieval_successful=False,
        hallucination_flagged=False
    )

    result = await agent.ainvoke(initial_state)

    return {
        "answer": result["answer"],
        "docs": result["documents"],
        "retrieval_attempted": result["retrieval_attempted"],
        "retrieval_successful": result["retrieval_successful"],
        "hallucination_flagged": result["hallucination_flagged"]
    }