

import streamlit as st
import numpy as np
import faiss
import torch

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# -------------------------------
# 1. Configure the application
# -------------------------------

st.set_page_config(
    page_title="Chat with Your PDF",
    page_icon="📚",
    layout="wide"
)

st.title("📚 Chat with Your PDF")
st.write("Ask questions about your PDF using a local RAG pipeline.")

device = "cuda" if torch.cuda.is_available() else "cpu"


# -------------------------------
# 2. Load AI models
# -------------------------------

@st.cache_resource
def load_models():
    embedding_model = SentenceTransformer(
        "sentence-transformers/all-MiniLM-L6-v2"
    )

    model_name = "google/flan-t5-base"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    llm = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    llm = llm.to(device)

    return embedding_model, tokenizer, llm


# -------------------------------
# 3. Extract PDF text
# -------------------------------

def extract_pdf(uploaded_file):
    reader = PdfReader(uploaded_file)

    chunks = []
    chunk_size = 500
    overlap = 50

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        start = 0

        while start < len(page_text):
            end = min(start + chunk_size, len(page_text))
            chunk_text = page_text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            if end == len(page_text):
                break

            start = end - overlap

    return chunks


# -------------------------------
# 4. Create the FAISS index
# -------------------------------

def create_index(chunks, embedding_model):
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    vectors = np.asarray(embeddings, dtype=np.float32)

    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)

    return index


# -------------------------------
# 5. Retrieve relevant chunks
# -------------------------------

def retrieve_chunks(question, chunks, index, embedding_model, top_k=3):
    question_embedding = embedding_model.encode(
        [question],
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    question_embedding = np.asarray(
        question_embedding,
        dtype=np.float32
    )

    k = min(top_k, index.ntotal)

    if k == 0:
        return []

    scores, indices = index.search(question_embedding, k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score)
            })

    return results


# -------------------------------
# 6. Generate an answer
# -------------------------------

def generate_answer(question, results, tokenizer, llm):
    if not results:
        return "No relevant information was found.", []

    context = "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}"
        for item in results
    )

    prompt = f"""
Answer the question using only the provided context.
If the answer is not available, say:
"I could not find the answer in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.no_grad():
        output = llm.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

    answer = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    source_pages = sorted({
        item["page"] for item in results
    })

    return answer, source_pages


# -------------------------------
# 7. Upload the PDF
# -------------------------------

uploaded_file = st.file_uploader(
    "Upload your PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    try:
        with st.spinner("Loading AI models..."):
            embedding_model, tokenizer, llm = load_models()

        with st.spinner("Reading your PDF..."):
            chunks = extract_pdf(uploaded_file)

        if not chunks:
            st.error(
                "No readable text was found. "
                "The PDF may contain scanned images."
            )
            st.stop()

        with st.spinner("Creating the search index..."):
            index = create_index(chunks, embedding_model)

        st.success(
            f"PDF processed successfully! "
            f"Created {len(chunks)} text chunks."
        )

        st.divider()
        st.subheader("Ask a question")

        question = st.text_input(
            "Enter your question about the PDF:"
        )

        if st.button("Get Answer"):
            if not question.strip():
                st.warning("Please enter a question.")
            else:
                with st.spinner("Finding information and generating an answer..."):
                    results = retrieve_chunks(
                        question,
                        chunks,
                        index,
                        embedding_model
                    )

                    answer, source_pages = generate_answer(
                        question,
                        results,
                        tokenizer,
                        llm
                    )

                st.subheader("Answer")
                st.write(answer)

                st.subheader("Source pages")

                if source_pages:
                    st.write(", ".join(map(str, source_pages)))
                else:
                    st.write("No source pages available.")

                with st.expander("View retrieved text"):
                    for item in results:
                        st.markdown(
                            f"**Page {item['page']}** "
                            f"(similarity: {item['score']:.3f})"
                        )
                        st.write(item["text"])
                        st.divider()

    except Exception as error:
        st.error(f"Something went wrong: {error}")