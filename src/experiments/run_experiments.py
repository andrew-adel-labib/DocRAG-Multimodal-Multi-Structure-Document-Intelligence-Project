import pandas as pd
from src.utils.logger import get_logger

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
        return pd.DataFrame(columns=["chunking", "embedding", "accuracy"])

    required_cols = ["Questions", "Golden Answers"]
    for col in required_cols:
        if col not in df.columns:
            logger.error(f"Missing column: {col}")
            return pd.DataFrame(columns=["chunking", "embedding", "accuracy"])

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
                    texts = [c["text"] for c in chunks if "text" in c]
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
                    rag = rag_builder(texts, embeddings, embedder)
                except Exception as e:
                    logger.error(f"RAG build failed: {e}")
                    continue

                correct = 0
                total = len(df)

                for _, row in df.iterrows():
                    q = str(row["Questions"])
                    truth = str(row["Golden Answers"])

                    try:
                        pred = rag.answer(q)

                        if not pred:
                            continue

                        pred = str(pred)

                        if truth.lower() in pred.lower():
                            correct += 1

                    except Exception as e:
                        logger.warning(f"Query failed: {e}")
                        continue

                acc = correct / total if total > 0 else 0

                results.append({
                    "chunking": chunk_name,
                    "embedding": getattr(embedder, "model_name", "unknown"),
                    "accuracy": acc
                })

                logger.info(f"{chunk_name} + {embedder.model_name}: {acc}")

            except Exception as e:
                logger.error(f"Experiment failed: {e}")
                continue

    if not results:
        logger.warning("No results generated")
        return pd.DataFrame(columns=["chunking", "embedding", "accuracy"])

    return pd.DataFrame(results)