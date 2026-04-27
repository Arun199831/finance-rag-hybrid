from typing import TypedDict, List, Any
from langgraph.graph import StateGraph, END


class AgentState(TypedDict):
    question: str
    documents: List[Any]
    answer: str
    retrieval_successful: bool


class AgentNodes:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm

    def retrieve_node(self, state: AgentState):
        try:
            docs = self.retriever.invoke(state["question"])

            return {
                "documents": docs,
                "retrieval_successful": len(docs) > 0
            }

        except Exception:
            return {
                "documents": [],
                "retrieval_successful": False
            }

    def generate_node(self, state: AgentState):
        docs = state["documents"]

        if not docs:
            return {
                "answer": "I don't have enough information."
            }

        prompt = f"""
Answer the question based only on the retrieved documents.

Question:
{state["question"]}

Documents:
{docs}
"""

        response = self.llm.invoke(prompt)

        answer = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        return {"answer": answer}

    def fallback_node(self, state: AgentState):
        return {
            "answer": "I don't have enough information to answer this question."
        }


def route_after_retrieval(state: AgentState):
    if state["retrieval_successful"]:
        return "generate"
    return "fallback"


def build_graph(retriever, llm):
    nodes = AgentNodes(retriever, llm)

    graph = StateGraph(AgentState)

    graph.add_node("retrieve", nodes.retrieve_node)
    graph.add_node("generate", nodes.generate_node)
    graph.add_node("fallback", nodes.fallback_node)

    graph.set_entry_point("retrieve")

    graph.add_conditional_edges(
        "retrieve",
        route_after_retrieval,
        {
            "generate": "generate",
            "fallback": "fallback"
        }
    )

    graph.add_edge("generate", END)
    graph.add_edge("fallback", END)

    return graph.compile()