import json
import os
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# ── Load eval dataset ──────────────────────────────────────────

def load_eval_dataset(path: str = "data/eval_dataset.json") -> list[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Eval dataset not found at {path}. "
            "Create data/eval_dataset.json first."
        )
    with open(path, "r") as f:
        return json.load(f)


# ── Run RAGAs evaluation ───────────────────────────────────────

async def run_eval(retriever) -> dict:
    samples = load_eval_dataset()

    questions = []
    answers = []
    contexts = []
    ground_truths = []

    for sample in samples:
        question = sample["question"]
        ground_truth = sample["ground_truth"]

        # retrieve contexts
        docs = await retriever.ainvoke(question)
        context_texts = [doc.page_content for doc in docs]

        # generate answer from context
        from app.chain import build_rag_chain
        chain = build_rag_chain(retriever)
        answer = await chain.ainvoke(question)

        questions.append(question)
        answers.append(answer)
        contexts.append(context_texts)
        ground_truths.append(ground_truth)

    # build RAGAs dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    # run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall,
        ],
        llm=ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        ),
        embeddings=OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )
    )

    return {
        "faithfulness": round(float(result["faithfulness"]), 4),
        "answer_relevancy": round(float(result["answer_relevancy"]), 4),
        "context_precision": round(float(result["context_precision"]), 4),
        "context_recall": round(float(result["context_recall"]), 4),
        "retriever_mode": "hybrid",
        "sample_size": len(samples)
    }