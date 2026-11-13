import os
from typing import List, Dict, Tuple

import faiss
import numpy as np
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


st.set_page_config(
    page_title="Intelligent HR RAG Assistant",
    page_icon="🤖",
    layout="wide",
)

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GROQ_MODEL = "openai/gpt-oss-120b"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150


@st.cache_resource(show_spinner="Loading Hugging Face embedding model...")
def load_embedding_model():
    return SentenceTransformer(MODEL_NAME)


def extract_pdf_text(uploaded_files) -> List[Dict[str, str]]:
    documents = []
    for uploaded_file in uploaded_files:
        reader = PdfReader(uploaded_file)
        page_texts = []
        for page in reader.pages:
            text = page.extract_text()
            if text:
                page_texts.append(text)
        documents.append({
            "source": uploaded_file.name,
            "text": "\n".join(page_texts),
        })
    return documents


def split_into_chunks(documents: List[Dict[str, str]]) -> List[Dict[str, str]]:
    chunks = []
    for doc in documents:
        text = doc["text"]
        start = 0
        while start < len(text):
            end = min(start + CHUNK_SIZE, len(text))
            if end < len(text):
                boundary = text.rfind(" ", start, end)
                if boundary > start:
                    end = boundary

            chunk_text = text[start:end]
            if chunk_text.strip():
                chunks.append({
                    "source": doc["source"],
                    "text": chunk_text,
                })

            if end == len(text):
                break

            start = end - CHUNK_OVERLAP
            while start < len(text) and not text[start].isspace():
                start += 1
            while start < len(text) and text[start].isspace():
                start += 1

    return chunks


def build_faiss_index(chunks, embedding_model):
    chunk_texts = [chunk["text"] for chunk in chunks]
    vectors = embedding_model.encode(
        chunk_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    vectors = np.asarray(vectors, dtype="float32")
    dimension = vectors.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(vectors)
    return index


def get_groq_api_key():
    try:
        return st.secrets["GROQ_API_KEY"]
    except Exception:
        return os.getenv("GROQ_API_KEY")


def answer_question(question: str, top_k: int):
    if "rag_index" not in st.session_state or "rag_chunks" not in st.session_state:
        st.warning("Upload and process your HR PDF documents first.")
        return

    api_key = get_groq_api_key()
    if not api_key:
        st.error(
            "Groq API key not found. Add GROQ_API_KEY in Streamlit "
            "Cloud app Secrets or in your local .streamlit/secrets.toml file."
        )
        return

    embedding_model = load_embedding_model()
    query_vector = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    query_vector = np.asarray(query_vector, dtype="float32")

    index = st.session_state["rag_index"]
    chunks = st.session_state["rag_chunks"]
    k = min(top_k, len(chunks))
    scores, indices = index.search(query_vector, k)

    retrieved = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            retrieved.append({
                "source": chunks[idx]["source"],
                "text": chunks[idx]["text"],
                "score": float(score),
            })

    context = "\n\n".join(
        f"Source: {item['source']}\n{item['text']}" for item in retrieved
    )

    client = Groq(api_key=api_key)
    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are an HR assistant. Answer using only the provided "
                    "HR policy documents. Give clear, accurate, simple answers. "
                    "Do not invent policy details. If the answer is not present "
                    "in the context, say: 'Information not found in the provided "
                    "documents.'"
                ),
            },
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ],
        temperature=0.2,
        max_tokens=300,
    )

    return response.choices[0].message.content, retrieved


st.title("🤖 Intelligent HR RAG Assistant")
st.caption(
    "Ask questions about company HR policies. Answers are generated from "
    "the PDF documents you upload."
)

with st.sidebar:
    st.header("1. Upload HR documents")
    uploaded_files = st.file_uploader(
        "Select one or more HR policy PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    st.caption("Upload the 11 HR PDF documents used in your Kaggle project.")

    st.header("2. Retrieval settings")
    top_k = st.slider("Relevant chunks to retrieve", min_value=1, max_value=5, value=3)

    process_clicked = st.button(
        "Process Documents",
        type="primary",
        use_container_width=True,
    )

    st.divider()
    st.markdown("**Technology stack**")
    st.markdown(
        "- PyPDF for PDF extraction\n"
        "- 1000-character chunks, 150 overlap\n"
        "- Hugging Face all-MiniLM-L6-v2\n"
        "- FAISS similarity search\n"
        "- Groq `openai/gpt-oss-120b`"
    )

if process_clicked:
    if not uploaded_files:
        st.sidebar.error("Please upload at least one PDF.")
    else:
        try:
            with st.spinner("Extracting text from PDF documents..."):
                documents = extract_pdf_text(uploaded_files)

            empty_docs = [d["source"] for d in documents if not d["text"].strip()]
            if empty_docs:
                st.warning(
                    "No selectable text was extracted from: "
                    + ", ".join(empty_docs)
                    + ". These may be scanned PDFs and need OCR."
                )

            documents = [d for d in documents if d["text"].strip()]
            if not documents:
                st.error("No text could be extracted from the uploaded PDFs.")
            else:
                with st.spinner("Splitting documents into chunks..."):
                    chunks = split_into_chunks(documents)

                with st.spinner("Creating embeddings and FAISS index..."):
                    embedding_model = load_embedding_model()
                    index = build_faiss_index(chunks, embedding_model)

                st.session_state["rag_documents"] = documents
                st.session_state["rag_chunks"] = chunks
                st.session_state["rag_index"] = index
                st.session_state["chat_history"] = []

                st.success(
                    f"Processed {len(documents)} documents and created "
                    f"{len(chunks)} chunks ({index.ntotal} FAISS vectors)."
                )
        except Exception as exc:
            st.exception(exc)

if "rag_index" in st.session_state:
    st.success(
        f"Knowledge base ready: {len(st.session_state['rag_documents'])} PDFs · "
        f"{len(st.session_state['rag_chunks'])} chunks · "
        f"{st.session_state['rag_index'].ntotal} vectors"
    )

    for message in st.session_state.get("chat_history", []):
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if message.get("sources"):
                with st.expander("Retrieved sources"):
                    for source in message["sources"]:
                        st.markdown(
                            f"**{source['source']}** · similarity "
                            f"{source['score']:.4f}"
                        )
                        st.write(source["text"][:700])
else:
    st.info("Upload your HR PDFs in the sidebar and click **Process Documents**.")

question = st.chat_input("Ask an HR policy question...")
if question:
    st.session_state.setdefault("chat_history", [])
    st.session_state["chat_history"].append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching HR documents and generating an answer..."):
            try:
                result = answer_question(question, top_k)
                if result:
                    answer, sources = result
                    st.markdown(answer)
                    with st.expander("Retrieved sources"):
                        for source in sources:
                            st.markdown(
                                f"**{source['source']}** · similarity "
                                f"{source['score']:.4f}"
                            )
                            st.write(source["text"][:700])

                    st.session_state["chat_history"].append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                    })
            except Exception as exc:
                st.error(f"Could not generate an answer: {exc}")
