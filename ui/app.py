import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import numpy as np
from dotenv import load_dotenv

from src.parser.advanced_parser import parse_pdf_advanced
from src.chunking.recursive_chunk import recursive_chunk
from src.chunking.sentence_chunk import sentence_chunk
from src.chunking.semantic_chunk import semantic_chunk
from src.chunking.late_chunk import late_chunk

from src.embedding.embedder import Embedder
from src.embedding.openai_embedder import OpenAIEmbedder
from src.embedding.clip_embedder import CLIPEmbedder

from src.retrieval.hybrid import HybridRetriever
from src.retrieval.reranker import Reranker
from src.retrieval.image_retriever import ImageRetriever

from src.query.rewriting import QueryRewriter
from src.rag.pipeline_advanced import AdvancedRAG

load_dotenv()

st.set_page_config(page_title="RAG System UI", layout="wide")
st.title("🚀 Multimodal RAG System")

st.sidebar.header("⚙️ Configuration")

chunking_option = st.sidebar.selectbox(
    "Chunking Method",
    ["recursive", "sentence", "semantic", "late"]
)

embedding_option = st.sidebar.selectbox(
    "Embedding Model",
    [
        "local-MiniLM",
        "bge-base",
        "bge-large",
        "openai-small",
        "openai-large"
    ]
)

use_reranker = st.sidebar.checkbox("Use Reranker", True)
use_rewrite = st.sidebar.checkbox("Use Query Rewriting", True)



def similarity_score(a, b, embedder):
    emb = embedder.encode([a, b])

    v1 = np.array(emb[0])
    v2 = np.array(emb[1])

    v1 = v1 / np.linalg.norm(v1)
    v2 = v2 / np.linalg.norm(v2)

    return float(np.dot(v1, v2))


@st.cache_resource
def load_golden(_embedder):
    df = pd.read_excel("data/golden_answers.xlsx")

    questions = df["Questions"].astype(str).tolist()
    answers = df["Golden Answers"].astype(str).tolist()

    embeddings = embedder.encode(questions)

    return df, questions, answers, embeddings


def find_best_match(query, questions, embeddings, embedder):
    q_emb = embedder.encode([query])[0]

    embeddings = np.array(embeddings)
    q_emb = np.array(q_emb)

    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    q_emb = q_emb / np.linalg.norm(q_emb)

    scores = np.dot(embeddings, q_emb.T)

    idx = int(np.argmax(scores))
    return idx, float(scores[idx])



@st.cache_resource
def build_pipeline(file_bytes, chunking_option, embedding_option):

    with open("temp.pdf", "wb") as f:
        f.write(file_bytes)

    docs = parse_pdf_advanced("temp.pdf")

    if chunking_option == "recursive":
        chunks = recursive_chunk(docs)
    elif chunking_option == "sentence":
        chunks = sentence_chunk(docs)
    elif chunking_option == "semantic":
        embedder_temp = Embedder("BAAI/bge-base-en-v1.5")
        chunks = semantic_chunk(docs, embedder_temp)
    else:
        embedder_temp = Embedder("BAAI/bge-base-en-v1.5")
        chunks = late_chunk(docs, embedder_temp)

    texts = [
        c["text"].strip()
        for c in chunks
        if c.get("text") and c["text"].strip()
    ]

    if embedding_option == "local-MiniLM":
        embedder = Embedder("all-MiniLM-L6-v2")
    elif embedding_option == "bge-base":
        embedder = Embedder("BAAI/bge-base-en-v1.5")
    elif embedding_option == "bge-large":
        embedder = Embedder("BAAI/bge-large-en-v1.5")
    elif embedding_option == "openai-small":
        embedder = OpenAIEmbedder("text-embedding-3-small")
    else:
        embedder = OpenAIEmbedder("text-embedding-3-large")

    embeddings = embedder.encode(texts)

    retriever = HybridRetriever(texts, embeddings)

    return docs, texts, retriever, embedder



uploaded_file = st.file_uploader("📄 Upload PDF", type=["pdf"])

if uploaded_file:

    file_bytes = uploaded_file.read()

    with st.spinner("Processing PDF..."):
        docs, texts, retriever, embedder = build_pipeline(
            file_bytes, chunking_option, embedding_option
        )

    st.success("✅ RAG Ready")
    st.info(f"📄 Pages: {len(docs)}")
    st.info(f"🧩 Chunks: {len(texts)}")

    @st.cache_resource
    def get_clip():
        return CLIPEmbedder()

    clip = get_clip()
    image_retriever = ImageRetriever(docs)

    reranker = Reranker() if use_reranker else None
    rewriter = QueryRewriter() if use_rewrite else None

    rag = AdvancedRAG(
        retriever,
        image_retriever,
        clip,
        reranker,
        rewriter,
        embedder
    )


    st.header("💬 Ask")
    query = st.text_input("Your question")

    if st.button("Get Answer"):
        answer = rag.answer(query)

        st.success("Answer:")
        st.write(answer)

        try:
            df, questions, answers, q_embeddings = load_golden(embedder)

            idx, q_score = find_best_match(query, questions, q_embeddings, embedder)

            matched_q = questions[idx]
            truth = answers[idx]

            pred_clean = answer.split("—")[0].strip()

            ans_score = similarity_score(pred_clean, truth, embedder)

            st.subheader("📊 Evaluation")

            st.write("🔍 Matched Question:")
            st.write(matched_q)
            st.write(f"Question Similarity: {q_score:.2f}")

            col1, col2 = st.columns(2)

            with col1:
                st.write("✅ Expected:")
                st.write(truth)

            with col2:
                st.write("🤖 Predicted:")
                st.write(answer)

            accuracy = ans_score * 100

            st.progress(min(accuracy / 100, 1.0))
            st.write(f"🎯 Accuracy: {accuracy:.2f}%")

            if ans_score >= 0.85:
                st.success("✔ Correct")
            elif ans_score >= 0.65:
                st.warning("⚠️ Partially Correct")
            else:
                st.error("Incorrect")

        except Exception as e:
            st.warning(f"Evaluation failed: {e}")

        if rag.is_visual_query(query):
            images = image_retriever.retrieve(query, clip)

            # if images:
            #     st.subheader("🖼 Images")
            #     for img in images:
            #         st.image(img["path"], width=200)


    st.header("📊 Experiments")

    if st.button("Run Experiments"):

        with st.spinner("Running experiments..."):

            try:
                df, questions, answers, q_embeddings = load_golden(embedder)

                records = []

                for q, truth in zip(questions, answers):
                    try:
                        pred = rag.answer(q)
                        pred_clean = pred.split("—")[0].strip()

                        score = similarity_score(pred_clean, truth, embedder)

                        records.append({
                            "question": q,
                            "truth": truth,
                            "prediction": pred,
                            "accuracy": round(score * 100, 2)
                        })

                    except:
                        continue

                results = pd.DataFrame(records)

                if results.empty:
                    st.warning("No results generated ⚠️")
                else:
                    st.success("Experiments completed")

                    st.dataframe(results)

                    avg_acc = results["accuracy"].mean()

                    st.subheader("📊 Overall Accuracy")
                    st.progress(min(avg_acc / 100, 1.0))
                    st.write(f"Average Accuracy: {avg_acc:.2f}%")

                    st.bar_chart(results.set_index("question")["accuracy"])

            except Exception as e:
                st.error(f"Experiment failed: {e}")