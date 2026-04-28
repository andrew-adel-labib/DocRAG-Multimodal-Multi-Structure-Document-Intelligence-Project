import os
from pathlib import Path
import pandas as pd
from dotenv import load_dotenv

from src.parser.advanced_parser import parse_pdf_advanced

from src.chunking.recursive_chunk import recursive_chunk
from src.chunking.sentence_chunk import sentence_chunk
from src.chunking.semantic_chunk import semantic_chunk
from src.chunking.late_chunk import late_chunk
from src.chunking.parent_child_chunk import parent_child_chunk
from src.chunking.section_aware_chunk import section_aware_chunk
from src.chunking.section_parent_chunk import section_parent_chunk
from src.chunking.raptor_chunk import raptor_chunk
from src.chunking.llm_chunk import llm_chunk

from src.embedding.embedder import Embedder
from src.embedding.openai_embedder import OpenAIEmbedder
from src.embedding.clip_embedder import CLIPEmbedder

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.image_retriever import ImageRetriever
from src.retrieval.reranker import Reranker

from src.query.rewriting import QueryRewriter
from src.rag.pipeline_advanced import AdvancedRAG

from src.evaluation.full_evaluator import (
    semantic_similarity,
    rouge_l_score,
    exact_match_score,
    compute_ragas,
    retrieval_metrics
)

load_dotenv()


rewriting_options = [False, True]
reranker_options = [False, True]

chunking_methods = {
    "recursive": recursive_chunk,
    "sentence": sentence_chunk,
    "semantic": semantic_chunk,
    "late": late_chunk,
    "parent-child": parent_child_chunk,
    "section-aware": section_aware_chunk,
    "section-parent": section_parent_chunk,
    "raptor": raptor_chunk,
    "llm": llm_chunk
}

embedding_models = {
    "MiniLM": lambda: Embedder("all-MiniLM-L6-v2"),
    "BGE Base": lambda: Embedder("BAAI/bge-base-en-v1.5"),
    "BGE Large": lambda: Embedder("BAAI/bge-large-en-v1.5"),
    "OpenAI Small": lambda: OpenAIEmbedder("text-embedding-3-small"),
    "OpenAI Large": lambda: OpenAIEmbedder("text-embedding-3-large")
}


def generate_experiment_calls():
    calls = []
    exp_id = 1

    for rewrite in rewriting_options:
        for rerank in reranker_options:
            for chunk_name, chunk_func in chunking_methods.items():
                for emb_name, emb_model in embedding_models.items():

                    calls.append({
                        "experiment_id": exp_id,
                        "rewrite": rewrite,
                        "rerank": rerank,
                        "chunking_name": chunk_name,
                        "chunk_func": chunk_func,
                        "embedding_name": emb_name,
                        "embedding_func": emb_model
                    })

                    exp_id += 1

    return calls


def run_all_experiments():

    print("🚀 Loading documents...")
    docs = parse_pdf_advanced("data/OneBEP.pdf")

    print("📄 Loading golden answers...")
    df_eval = pd.read_excel("data/golden_answers.xlsx")

    Path("data/experiment_results").mkdir(
        parents=True,
        exist_ok=True
    )

    Path("data/experiment_details").mkdir(
        parents=True,
        exist_ok=True
    )

    all_results = []

    experiment_calls = generate_experiment_calls()

    for config in experiment_calls:

        print(
            f"\n🧪 Running Experiment {config['experiment_id']} | "
            f"Rewrite={config['rewrite']} | "
            f"Rerank={config['rerank']} | "
            f"Chunk={config['chunking_name']} | "
            f"Embed={config['embedding_name']}"
        )

        try:

            if (
                "OpenAI" in config["embedding_name"]
                and not os.getenv("OPENAI_API_KEY")
            ):
                print(
                    "⚠ Skipping OpenAI experiment (missing API key)"
                )
                continue

            # Optional skip heavy LLM chunking bulk mode
            # if config["chunking_name"] == "llm":
            #     continue

            embedder = config["embedding_func"]()

            if config["chunking_name"] in [
                "semantic",
                "late",
                "raptor"
            ]:
                chunks = config["chunk_func"](
                    docs,
                    embedder
                )

            elif config["chunking_name"] in [
                "recursive",
                "sentence",
                "parent-child",
                "section-aware",
                "section-parent",
                "llm"
            ]:
                chunks = config["chunk_func"](docs)

            else:
                chunks = recursive_chunk(docs)

            texts = [
                c["text"]
                for c in chunks
                if (
                    "text" in c
                    and c["text"]
                    and c["text"].strip()
                )
            ]

            if not texts:
                print("⚠ No valid chunks")
                continue

            print(f"📦 Chunk count: {len(texts)}")

            embeddings = embedder.encode(texts)

            
            retriever = HybridRetriever(
                texts,
                embeddings
            )

            reranker = (
                Reranker()
                if config["rerank"]
                else None
            )

            rewriter = (
                QueryRewriter()
                if config["rewrite"]
                else None
            )

            rag = AdvancedRAG(
                retriever=retriever,
                image_retriever=ImageRetriever(docs),
                clip=CLIPEmbedder(),
                reranker=reranker,
                rewriter=rewriter,
                embedder=embedder
            )

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

            detailed_results = []

            for idx, row in df_eval.iterrows():

                question = str(
                    row["Questions"]
                ).strip()

                truth = str(
                    row["Golden Answers"]
                ).strip()

                if not question or not truth:
                    continue

                try:

                    retrieved_docs = retriever.retrieve(
                        question,
                        embedder,
                        k=10
                    )

                    pred = rag.answer(question)

                    if not pred:
                        continue

                    print(f"Q: {question}")
                    print(f"PRED: {pred}")
                    print(f"TRUTH: {truth}")

                    safe_docs = [
                        doc[:1000]
                        for doc in retrieved_docs[:3]
                    ]

                    pred_safe = str(pred)[:1500]
                    truth_safe = str(truth)[:1500]

                    sem_score = semantic_similarity(
                        embedder,
                        pred,
                        truth
                    )

                    rouge_score = rouge_l_score(
                        pred,
                        truth
                    )

                    exact_score = exact_match_score(
                        pred,
                        truth
                    )

                    semantic_scores.append(
                        sem_score
                    )

                    rouge_scores.append(
                        rouge_score
                    )

                    exact_scores.append(
                        exact_score
                    )

                    if len(truth.split()) > 150:
                        ragas_scores = {
                            "faithfulness": 0.0,
                            "answer_correctness": 0.0,
                            "context_precision": 0.0,
                            "context_recall": 0.0
                        }

                    else:
                        ragas_scores = compute_ragas(
                            question,
                            pred_safe,
                            truth_safe,
                            safe_docs
                        )

                    faithfulness_scores.append(
                        ragas_scores.get(
                            "faithfulness",
                            0.0
                        )
                    )

                    answer_correctness_scores.append(
                        ragas_scores.get(
                            "answer_correctness",
                            0.0
                        )
                    )

                    context_precision_scores.append(
                        ragas_scores.get(
                            "context_precision",
                            0.0
                        )
                    )

                    context_recall_scores.append(
                        ragas_scores.get(
                            "context_recall",
                            0.0
                        )
                    )

                    retrieved_map = {
                        f"doc_{i}": 1 / (i + 1)
                        for i, _ in enumerate(
                            retrieved_docs
                        )
                    }

                    relevant_docs = ["doc_0"]

                    retrieval_scores = retrieval_metrics(
                        query_id=str(idx),
                        retrieved_docs=retrieved_map,
                        relevant_docs=relevant_docs
                    )

                    ndcg_scores.append(
                        retrieval_scores["ndcg@10"]
                    )

                    recall_scores.append(
                        retrieval_scores["recall@10"]
                    )

                    mrr_scores.append(
                        retrieval_scores["mrr"]
                    )

                    detailed_results.append({
                        "question_id": idx,
                        "question": question,
                        "prediction": pred,
                        "ground_truth": truth,

                        "semantic_similarity": sem_score,
                        "rouge_l": rouge_score,
                        "exact_match": exact_score,

                        "faithfulness": ragas_scores.get(
                            "faithfulness",
                            0.0
                        ),

                        "answer_correctness": ragas_scores.get(
                            "answer_correctness",
                            0.0
                        ),

                        "context_precision": ragas_scores.get(
                            "context_precision",
                            0.0
                        ),

                        "context_recall": ragas_scores.get(
                            "context_recall",
                            0.0
                        ),

                        "ndcg@10": retrieval_scores.get(
                            "ndcg@10",
                            0.0
                        ),

                        "recall@10": retrieval_scores.get(
                            "recall@10",
                            0.0
                        ),

                        "mrr": retrieval_scores.get(
                            "mrr",
                            0.0
                        )
                    })

                except Exception as e:
                    print(
                        f"⚠ Question failed: {e}"
                    )
                    continue

            result = {
                "experiment_id": config["experiment_id"],
                "query_rewriting": config["rewrite"],
                "reranker": config["rerank"],
                "chunking_method": config["chunking_name"],
                "embedding_model": config["embedding_name"],

                "semantic_similarity":
                    sum(semantic_scores) / len(semantic_scores)
                    if semantic_scores else 0.0,

                "rouge_l":
                    sum(rouge_scores) / len(rouge_scores)
                    if rouge_scores else 0.0,

                "exact_match":
                    sum(exact_scores) / len(exact_scores)
                    if exact_scores else 0.0,

                "faithfulness":
                    sum(faithfulness_scores) / len(faithfulness_scores)
                    if faithfulness_scores else 0.0,

                "answer_correctness":
                    sum(answer_correctness_scores) / len(answer_correctness_scores)
                    if answer_correctness_scores else 0.0,

                "context_precision":
                    sum(context_precision_scores) / len(context_precision_scores)
                    if context_precision_scores else 0.0,

                "context_recall":
                    sum(context_recall_scores) / len(context_recall_scores)
                    if context_recall_scores else 0.0,

                "ndcg@10":
                    sum(ndcg_scores) / len(ndcg_scores)
                    if ndcg_scores else 0.0,

                "recall@10":
                    sum(recall_scores) / len(recall_scores)
                    if recall_scores else 0.0,

                "mrr":
                    sum(mrr_scores) / len(mrr_scores)
                    if mrr_scores else 0.0,
            }

            all_results.append(result)

            pd.DataFrame(
                all_results
            ).to_csv(
                "data/full_experiment_results_partial.csv",
                index=False
            )

            pd.DataFrame(
                [result]
            ).to_csv(
                f"data/experiment_results/experiment_{config['experiment_id']}.csv",
                index=False
            )

            if detailed_results:
                pd.DataFrame(
                    detailed_results
                ).to_csv(
                    f"data/experiment_details/experiment_{config['experiment_id']}_details.csv",
                    index=False
                )

            print(
                f"✅ Experiment {config['experiment_id']} completed."
            )

        except Exception as e:
            print(
                f"❌ Experiment failed: {e}"
            )
            continue

    if not all_results:
        print(
            "❌ No successful experiments generated."
        )
        return pd.DataFrame()

    results_df = pd.DataFrame(
        all_results
    )

    results_df.to_csv(
        "data/full_experiment_results.csv",
        index=False
    )

    print("\nAll experiments completed.")
    print(
        "📊 Results saved to data/full_experiment_results.csv"
    )

    return results_df


if __name__ == "__main__":
    run_all_experiments()