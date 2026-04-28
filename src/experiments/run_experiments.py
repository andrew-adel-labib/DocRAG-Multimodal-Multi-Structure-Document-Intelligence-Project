import pandas as pd

from src.utils.logger import get_logger
from src.evaluation.full_evaluator import (
    semantic_similarity,
    rouge_l_score,
    exact_match_score,
    compute_ragas,
    retrieval_metrics
)

logger = get_logger("Experiments")


def run_all_experiments(
    docs,
    chunking_methods,
    embedding_models,
    rag_builder,
    eval_file
):
    results = []

    try:
        df = pd.read_excel(eval_file)
    except Exception as e:
        logger.error(f"Failed to load eval file: {e}")
        return pd.DataFrame()

    required_cols = ["Questions", "Golden Answers"]

    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing column: {col}")
            return pd.DataFrame()

    for chunk_name, chunk_func in chunking_methods.items():
        logger.info(f"Chunking: {chunk_name}")

        for emb_model in embedding_models:
            logger.info(f"Embedding: {emb_model}")

            try:
                embedder = emb_model()

                try:
                    chunks = chunk_func(docs)

                    if not chunks:
                        raise ValueError("Empty chunks")

                except Exception as e:
                    logger.error(f"Chunking failed: {e}")
                    continue

                try:
                    texts = [
                        c["text"]
                        for c in chunks
                        if "text" in c and c["text"] and c["text"].strip()
                    ]

                    if not texts:
                        raise ValueError("No valid texts")

                except Exception as e:
                    logger.error(f"Text extraction failed: {e}")
                    continue

                try:
                    embeddings = embedder.encode(texts)

                except Exception as e:
                    logger.error(f"Embedding failed: {e}")
                    continue

                try:
                    rag = rag_builder(
                        texts,
                        embeddings,
                        embedder
                    )

                except Exception as e:
                    logger.error(f"RAG build failed: {e}")
                    continue

                total = len(df)

                semantic_scores = []
                rouge_scores = []
                exact_scores = []

                faithfulness_scores = []
                answer_correctness_scores = []
                context_precision_scores = []
                context_recall_scores = []

                ndcg_scores = []
                recall_scores = []
                mrr_scores = []

                for idx, row in df.iterrows():

                    q = str(row["Questions"]).strip()
                    truth = str(row["Golden Answers"]).strip()

                    if not q or not truth:
                        continue

                    try:
                        retrieved_docs = rag.retriever.retrieve(
                            q,
                            embedder,
                            k=10
                        )

                        pred = rag.answer(q)

                        if not pred:
                            continue

                        pred = str(pred)

                        semantic = semantic_similarity(
                            embedder,
                            pred,
                            truth
                        )

                        semantic_scores.append(semantic)

                        rouge = rouge_l_score(
                            pred,
                            truth
                        )

                        rouge_scores.append(rouge)

                        exact = exact_match_score(
                            pred,
                            truth
                        )

                        exact_scores.append(exact)

                        ragas_scores = compute_ragas(
                            q,
                            pred,
                            truth,
                            retrieved_docs
                        )

                        faithfulness_scores.append(
                            ragas_scores.get("faithfulness", 0.0)
                        )

                        answer_correctness_scores.append(
                            ragas_scores.get("answer_correctness", 0.0)
                        )

                        context_precision_scores.append(
                            ragas_scores.get("context_precision", 0.0)
                        )

                        context_recall_scores.append(
                            ragas_scores.get("context_recall", 0.0)
                        )

                        retrieved_map = {
                            f"doc_{i}": 1 / (i + 1)
                            for i, _ in enumerate(retrieved_docs)
                        }

                        relevant_docs = ["doc_0"]

                        retrieval_scores = retrieval_metrics(
                            query_id=str(idx),
                            retrieved_docs=retrieved_map,
                            relevant_docs=relevant_docs
                        )

                        ndcg_scores.append(
                            retrieval_scores.get("ndcg@10", 0.0)
                        )

                        recall_scores.append(
                            retrieval_scores.get("recall@10", 0.0)
                        )

                        mrr_scores.append(
                            retrieval_scores.get("mrr", 0.0)
                        )

                    except Exception as e:
                        logger.warning(f"Query failed: {e}")
                        continue

                results.append({
                    "chunking": chunk_name,
                    "embedding": getattr(
                        embedder,
                        "model_name",
                        "unknown"
                    ),

                    "semantic_similarity": (
                        sum(semantic_scores) / len(semantic_scores)
                        if semantic_scores else 0.0
                    ),

                    "rouge_l": (
                        sum(rouge_scores) / len(rouge_scores)
                        if rouge_scores else 0.0
                    ),

                    "exact_match": (
                        sum(exact_scores) / len(exact_scores)
                        if exact_scores else 0.0
                    ),

                    "faithfulness": (
                        sum(faithfulness_scores) / len(faithfulness_scores)
                        if faithfulness_scores else 0.0
                    ),

                    "answer_correctness": (
                        sum(answer_correctness_scores) / len(answer_correctness_scores)
                        if answer_correctness_scores else 0.0
                    ),

                    "context_precision": (
                        sum(context_precision_scores) / len(context_precision_scores)
                        if context_precision_scores else 0.0
                    ),

                    "context_recall": (
                        sum(context_recall_scores) / len(context_recall_scores)
                        if context_recall_scores else 0.0
                    ),

                    "ndcg@10": (
                        sum(ndcg_scores) / len(ndcg_scores)
                        if ndcg_scores else 0.0
                    ),

                    "recall@10": (
                        sum(recall_scores) / len(recall_scores)
                        if recall_scores else 0.0
                    ),

                    "mrr": (
                        sum(mrr_scores) / len(mrr_scores)
                        if mrr_scores else 0.0
                    )
                })

                logger.info(
                    f"{chunk_name} + "
                    f"{getattr(embedder, 'model_name', 'unknown')} completed."
                )

            except Exception as e:
                logger.error(f"Experiment failed: {e}")
                continue

    if not results:
        logger.warning("No results generated")
        return pd.DataFrame()

    return pd.DataFrame(results)