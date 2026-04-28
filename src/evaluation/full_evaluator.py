import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from rouge_score import rouge_scorer
import evaluate as hf_evaluate

from ragas import evaluate as ragas_evaluate
from ragas.metrics import (
    faithfulness,
    answer_correctness,
    context_precision,
    context_recall
)

from datasets import Dataset
from ranx import Qrels, Run, evaluate as ranx_evaluate


exact_match = hf_evaluate.load("exact_match")


def semantic_similarity(embedder, pred, truth):
    try:
        if not pred or not truth:
            return 0.0

        pred = str(pred).strip()
        truth = str(truth).strip()

        if not pred or not truth:
            return 0.0

        emb = embedder.encode([pred, truth])

        return float(
            cosine_similarity(
                [emb[0]],
                [emb[1]]
            )[0][0]
        )

    except Exception:
        return 0.0


def rouge_l_score(pred, truth):
    try:
        if not pred or not truth:
            return 0.0

        scorer = rouge_scorer.RougeScorer(
            ["rougeL"],
            use_stemmer=True
        )

        scores = scorer.score(
            str(truth),
            str(pred)
        )

        return float(
            scores["rougeL"].fmeasure
        )

    except Exception:
        return 0.0


def exact_match_score(pred, truth):
    try:
        if not pred or not truth:
            return 0.0

        return exact_match.compute(
            predictions=[str(pred)],
            references=[str(truth)]
        )["exact_match"]

    except Exception:
        return 0.0


def compute_ragas(question, answer, truth, contexts):
    try:
        if not question or not answer or not truth:
            raise ValueError("Missing inputs")

        question = str(question).strip()[:1000]
        answer = str(answer).strip()[:1500]
        truth = str(truth).strip()[:1500]

        if len(truth.split()) > 150:
            return {
                "faithfulness": 0.0,
                "answer_correctness": 0.0,
                "context_precision": 0.0,
                "context_recall": 0.0
            }

        if not contexts:
            contexts = []

        cleaned_contexts = []

        for ctx in contexts[:3]:  # top 3 only
            if ctx and str(ctx).strip():
                cleaned_contexts.append(
                    str(ctx).strip()[:1000]
                )

        if not cleaned_contexts:
            cleaned_contexts = [""]

        dataset = Dataset.from_dict({
            "question": [question],
            "answer": [answer],
            "ground_truth": [truth],
            "contexts": [cleaned_contexts]
        })

        result = ragas_evaluate(
            dataset,
            metrics=[
                faithfulness,
                answer_correctness,
                context_precision,
                context_recall
            ]
        )

        result_dict = (
            result.to_pandas()
            .iloc[0]
            .to_dict()
        )

        return {
            "faithfulness": float(
                result_dict.get("faithfulness", 0.0)
            ),
            "answer_correctness": float(
                result_dict.get("answer_correctness", 0.0)
            ),
            "context_precision": float(
                result_dict.get("context_precision", 0.0)
            ),
            "context_recall": float(
                result_dict.get("context_recall", 0.0)
            )
        }

    except Exception:
        return {
            "faithfulness": 0.0,
            "answer_correctness": 0.0,
            "context_precision": 0.0,
            "context_recall": 0.0
        }


def retrieval_metrics(
    query_id,
    retrieved_docs,
    relevant_docs
):
    try:
        if not retrieved_docs:
            raise ValueError("No retrieved docs")

        if not relevant_docs:
            relevant_docs = []

        qrels = Qrels({
            str(query_id): {
                str(doc_id): 1
                for doc_id in relevant_docs
            }
        })

        run = Run({
            str(query_id): {
                str(doc_id): float(score)
                for doc_id, score in retrieved_docs.items()
            }
        })

        scores = ranx_evaluate(
            qrels=qrels,
            run=run,
            metrics=[
                "ndcg@10",
                "recall@10",
                "mrr"
            ]
        )

        return {
            "ndcg@10": float(
                scores.get("ndcg@10", 0.0)
            ),
            "recall@10": float(
                scores.get("recall@10", 0.0)
            ),
            "mrr": float(
                scores.get("mrr", 0.0)
            )
        }

    except Exception:
        return {
            "ndcg@10": 0.0,
            "recall@10": 0.0,
            "mrr": 0.0
        }