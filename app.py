%%writefile app.py

import io
import hashlib
import html

import streamlit as st
import numpy as np
import faiss
import torch

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# ============================================================
# 1. PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="DocuMind AI | PDF Intelligence",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. CUSTOM DESIGN SYSTEM
# ============================================================

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

    :root {
        --primary: #635BFF;
        --primary-dark: #4F46E5;
        --background: #F7F8FC;
        --surface: #FFFFFF;
        --text: #172033;
        --muted: #68738A;
        --border: #E5E9F2;
        --success: #16875D;
    }

    .stApp {
        background: var(--background);
        color: var(--text);
        font-family: 'DM Sans', sans-serif;
    }

    [data-testid="stHeader"] {
        background: rgba(247, 248, 252, 0.92);
    }

    [data-testid="stMainBlockContainer"] {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }

    h1, h2, h3 {
        font-family: 'Manrope', sans-serif !important;
        color: var(--text);
        letter-spacing: -0.6px;
    }

    .hero {
        padding: 34px 36px;
        border-radius: 24px;
        background: linear-gradient(120deg, #25245E 0%, #4F46E5 60%, #8275FF 100%);
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 14px 36px rgba(79, 70, 229, 0.15);
    }

    .hero-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 30px;
        background: rgba(255,255,255,0.14);
        border: 1px solid rgba(255,255,255,0.22);
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1px;
        margin-bottom: 14px;
    }

    .hero-title {
        color: white !important;
        font-size: clamp(28px, 4vw, 43px);
        line-height: 1.2;
        font-weight: 800;
        margin: 0 0 12px 0;
    }

    .hero-subtitle {
        color: rgba(255,255,255,0.83);
        font-size: 15px;
        line-height: 1.8;
        max-width: 760px;
        margin-bottom: 18px;
    }

    .hero-tag {
        display: inline-block;
        padding: 7px 11px;
        margin: 3px 5px 0 0;
        border-radius: 8px;
        background: rgba(255,255,255,0.12);
        color: white;
        font-size: 12px;
        font-weight: 600;
    }

    .section-label {
        color: #7A8499;
        text-transform: uppercase;
        font-size: 11px;
        font-weight: 800;
        letter-spacing: 1.5px;
        margin-bottom: 8px;
    }

    .section-heading {
        font-family: 'Manrope', sans-serif;
        font-size: 23px;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 5px;
    }

    .section-description {
        color: var(--muted);
        font-size: 14px;
        line-height: 1.7;
        margin-bottom: 18px;
    }

    .metric-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 19px;
        min-height: 112px;
        box-shadow: 0 3px 12px rgba(25, 35, 70, 0.025);
    }

    .metric-label {
        color: var(--muted);
        font-size: 12px;
        font-weight: 600;
        margin-bottom: 10px;
    }

    .metric-value {
        color: var(--text);
        font-family: 'Manrope', sans-serif;
        font-size: 25px;
        font-weight: 800;
    }

    .metric-caption {
        color: #929BAD;
        font-size: 11px;
        margin-top: 4px;
    }

    .panel {
        background: white;
        border: 1px solid var(--border);
        border-radius: 19px;
        padding: 23px;
        margin-bottom: 18px;
    }

    .assistant-card {
        background: white;
        border: 1px solid var(--border);
        border-radius: 17px;
        padding: 20px 22px;
        margin: 12px 0;
        line-height: 1.85;
        box-shadow: 0 3px 12px rgba(25, 35, 70, 0.025);
    }

    .question-card {
        background: #F0EFFF;
        border: 1px solid #DFDCFF;
        border-radius: 16px;
        padding: 17px 20px;
        margin: 12px 0;
        line-height: 1.7;
    }

    .answer-label {
        font-size: 11px;
        font-weight: 800;
        color: var(--primary-dark);
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 9px;
    }

    .source-card {
        background: #FAFBFE;
        border: 1px solid var(--border);
        border-radius: 13px;
        padding: 15px;
        margin: 10px 0;
        line-height: 1.75;
        font-size: 13px;
        color: #39445A;
    }

    .source-badge {
        display: inline-block;
        padding: 4px 9px;
        background: #ECEBFF;
        color: #5147D9;
        border-radius: 6px;
        font-size: 11px;
        font-weight: 800;
        margin-bottom: 8px;
    }

    .info-banner {
        background: #F0F0FF;
        border: 1px solid #DFDCFF;
        border-radius: 13px;
        padding: 15px 17px;
        color: #44418C;
        font-size: 13px;
        line-height: 1.7;
        margin: 12px 0;
    }

    .success-banner {
        background: #EAF8F1;
        border: 1px solid #CDEDDD;
        border-radius: 13px;
        padding: 15px 17px;
        color: #176B4B;
        font-size: 13px;
        line-height: 1.7;
        margin: 12px 0;
    }

    .sidebar-brand {
        font-family: 'Manrope', sans-serif;
        font-size: 23px;
        font-weight: 800;
        color: #635BFF;
        margin-bottom: 2px;
    }

    .sidebar-subtitle {
        font-size: 12px;
        color: #8791A5;
        margin-bottom: 25px;
    }

    .sidebar-section {
        font-size: 11px;
        color: #8992A5;
        text-transform: uppercase;
        letter-spacing: 1.3px;
        font-weight: 800;
        margin: 20px 0 10px 0;
    }

    .sidebar-model {
        padding: 12px;
        border: 1px solid #E5E9F2;
        border-radius: 11px;
        background: white;
        margin: 8px 0;
    }

    .sidebar-model-title {
        color: #303A50;
        font-size: 12px;
        font-weight: 700;
    }

    .sidebar-model-subtitle {
        color: #8490A5;
        font-size: 11px;
        margin-top: 4px;
        overflow-wrap: anywhere;
    }

    div.stButton > button {
        border-radius: 10px;
        font-weight: 700;
        border: 1px solid #E1E5EF;
        min-height: 42px;
        transition: all 0.2s ease;
    }

    div.stButton > button[kind="primary"] {
        background: var(--primary);
        border-color: var(--primary);
        color: white;
    }

    div.stButton > button[kind="primary"]:hover {
        background: var(--primary-dark);
        border-color: var(--primary-dark);
    }

    div[data-testid="stFileUploader"] {
        background: white;
        border: 1px dashed #C8CDE0;
        border-radius: 14px;
        padding: 10px;
    }

    div[data-testid="stTextInput"] input {
        border-radius: 11px;
    }

    div[data-testid="stExpander"] {
        background: white;
        border: 1px solid var(--border);
        border-radius: 12px;
    }

    .footer {
        text-align: center;
        color: #929BAD;
        font-size: 12px;
        padding: 25px 0 5px 0;
        border-top: 1px solid var(--border);
        margin-top: 35px;
    }

    @media (max-width: 768px) {
        [data-testid="stMainBlockContainer"] {
            padding: 1rem 0.8rem 2rem 0.8rem;
        }

        .hero {
            padding: 24px 21px;
            border-radius: 18px;
        }

        .panel {
            padding: 16px;
        }
    }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 3. APPLICATION SETTINGS
# ============================================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
LLM_MODEL_NAME = "google/flan-t5-base"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
TOP_K = 3


# ============================================================
# 4. SESSION STATE
# ============================================================

if "document_hash" not in st.session_state:
    st.session_state.document_hash = None

if "chunks" not in st.session_state:
    st.session_state.chunks = []

if "index" not in st.session_state:
    st.session_state.index = None

if "page_count" not in st.session_state:
    st.session_state.page_count = 0

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "document_name" not in st.session_state:
    st.session_state.document_name = None

if "processing_error" not in st.session_state:
    st.session_state.processing_error = None


# ============================================================
# 5. LOAD AI MODELS
# ============================================================

@st.cache_resource(show_spinner=False)
def load_models():
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL_NAME
    )

    tokenizer = AutoTokenizer.from_pretrained(
        LLM_MODEL_NAME
    )

    llm = AutoModelForSeq2SeqLM.from_pretrained(
        LLM_MODEL_NAME
    )

    llm = llm.to(DEVICE)
    llm.eval()

    return embedding_model, tokenizer, llm


# ============================================================
# 6. EXTRACT AND CHUNK PDF TEXT
# ============================================================

def extract_pdf(pdf_bytes):
    reader = PdfReader(io.BytesIO(pdf_bytes))

    chunks = []

    for page_number, page in enumerate(reader.pages, start=1):
        page_text = page.extract_text() or ""

        page_text = page_text.strip()

        if not page_text:
            continue

        start = 0

        while start < len(page_text):
            end = min(
                start + CHUNK_SIZE,
                len(page_text)
            )

            chunk_text = page_text[start:end]

            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "page": page_number
                })

            if end >= len(page_text):
                break

            start = end - CHUNK_OVERLAP

    return chunks, len(reader.pages)


# ============================================================
# 7. BUILD THE FAISS INDEX
# ============================================================

def create_index(chunks, embedding_model):
    if not chunks:
        raise ValueError(
            "No readable text was found in this PDF."
        )

    texts = [
        chunk["text"]
        for chunk in chunks
    ]

    embeddings = embedding_model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False
    )

    vectors = np.asarray(
        embeddings,
        dtype=np.float32
    )

    index = faiss.IndexFlatIP(
        vectors.shape[1]
    )

    index.add(vectors)

    return index


# ============================================================
# 8. RETRIEVE RELEVANT CHUNKS
# ============================================================

def retrieve_chunks(
    question,
    chunks,
    index,
    embedding_model,
    top_k=TOP_K
):
    if index is None or index.ntotal == 0:
        return []

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

    scores, indices = index.search(
        question_embedding,
        k
    )

    results = []

    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            results.append({
                "text": chunks[idx]["text"],
                "page": chunks[idx]["page"],
                "score": float(score)
            })

    return results


# ============================================================
# 9. GENERATE AN ANSWER
# ============================================================

def generate_answer(
    question,
    results,
    tokenizer,
    llm
):
    if not results:
        return (
            "I could not find relevant information "
            "in the provided document.",
            []
        )

    context = "\n\n".join(
        f"[Page {item['page']}]\n{item['text']}"
        for item in results
    )

    prompt = f"""
Answer the question using only the information in the context.

If the answer is not available in the context, say:
"I could not find the answer in the provided document."

Do not invent facts. Be clear and concise.

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
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    with torch.inference_mode():
        output = llm.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=False
        )

    answer = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    ).strip()

    source_pages = sorted({
        item["page"]
        for item in results
    })

    return answer, source_pages


# ============================================================
# 10. SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="sidebar-brand">✦ DocuMind AI</div>
        <div class="sidebar-subtitle">
            Intelligent document assistant
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="sidebar-section">Your document</div>',
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a PDF document",
        type=["pdf"],
        help="Choose a text-based PDF to start asking questions."
    )

    if uploaded_file is not None:
        st.caption(f"Selected: {uploaded_file.name}")
        st.caption(
            f"File size: {uploaded_file.size / (1024 * 1024):.2f} MB"
        )

    st.markdown(
        '<div class="sidebar-section">How it works</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        **01 · Upload**  
        Select your PDF document.

        **02 · Understand**  
        The AI extracts text and creates semantic embeddings.

        **03 · Retrieve**  
        FAISS finds the most relevant passages.

        **04 · Ask**  
        FLAN-T5 generates an answer using retrieved context.
        """
    )

    st.markdown(
        '<div class="sidebar-section">AI models</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        f"""
        <div class="sidebar-model">
            <div class="sidebar-model-title">Embedding model</div>
            <div class="sidebar-model-subtitle">
                {EMBEDDING_MODEL_NAME}
            </div>
        </div>

        <div class="sidebar-model">
            <div class="sidebar-model-title">Language model</div>
            <div class="sidebar-model-subtitle">
                {LLM_MODEL_NAME}
            </div>
        </div>

        <div class="sidebar-model">
            <div class="sidebar-model-title">Vector search</div>
            <div class="sidebar-model-subtitle">
                FAISS · Exact similarity search
            </div>
        </div>

        <div class="sidebar-model">
            <div class="sidebar-model-title">Execution device</div>
            <div class="sidebar-model-subtitle">
                {DEVICE.upper()}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.caption(
        "Built with Python, Streamlit, FAISS, "
        "Sentence Transformers and Hugging Face."
    )

    st.caption(
        "Privacy reminder: avoid uploading confidential "
        "documents to publicly exposed deployments."
    )


# ============================================================
# 11. MAIN HEADER
# ============================================================

st.markdown(
    """
    <div class="hero">
        <div class="hero-badge">AI-POWERED DOCUMENT INTELLIGENCE</div>

        <div class="hero-title">
            Your documents.<br>Understood.
        </div>

        <div class="hero-subtitle">
            Turn lengthy PDFs into a searchable knowledge source.
            Ask questions in plain English, explore relevant passages,
            and verify answers using their source pages.
        </div>

        <span class="hero-tag">✦ Retrieval-Augmented Generation</span>
        <span class="hero-tag">◈ Semantic Search</span>
        <span class="hero-tag">⌘ Local Model Inference</span>
    </div>
    """,
    unsafe_allow_html=True
)


# ============================================================
# 12. PROCESS THE UPLOADED DOCUMENT
# ============================================================

if uploaded_file is not None:

    pdf_bytes = uploaded_file.getvalue()

    current_hash = hashlib.sha256(
        pdf_bytes
    ).hexdigest()

    if current_hash != st.session_state.document_hash:

        # Reset old document data before processing a new PDF.
        st.session_state.document_hash = None
        st.session_state.chunks = []
        st.session_state.index = None
        st.session_state.page_count = 0
        st.session_state.chat_history = []
        st.session_state.document_name = uploaded_file.name
        st.session_state.processing_error = None

        try:
            if not pdf_bytes:
                raise ValueError(
                    "The uploaded PDF is empty."
                )

            with st.spinner(
                "Loading AI models and processing your document..."
            ):

                embedding_model, tokenizer, llm = load_models()

                chunks, page_count = extract_pdf(
                    pdf_bytes
                )

                if not chunks:
                    raise ValueError(
                        "No readable text was found. "
                        "The PDF may be scanned or image-based."
                    )

                index = create_index(
                    chunks,
                    embedding_model
                )

                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.page_count = page_count
                st.session_state.document_hash = current_hash
                st.session_state.document_name = uploaded_file.name

        except Exception as error:
            st.session_state.processing_error = str(error)


# ============================================================
# 13. SHOW PROCESSING ERRORS
# ============================================================

if st.session_state.processing_error:

    st.error(
        f"Document processing failed: "
        f"{st.session_state.processing_error}"
    )

    st.info(
        "Try another PDF. If this document is scanned, "
        "OCR support may be required."
    )


# ============================================================
# 14. DOCUMENT DASHBOARD
# ============================================================

if st.session_state.index is not None:

    st.markdown(
        '<div class="section-label">Workspace overview</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">Document dashboard</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Your document is indexed and ready for questions.'
        '</div>',
        unsafe_allow_html=True
    )

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">DOCUMENT PAGES</div>
                <div class="metric-value">
                    {st.session_state.page_count}
                </div>
                <div class="metric-caption">Total PDF pages</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">TEXT CHUNKS</div>
                <div class="metric-value">
                    {len(st.session_state.chunks)}
                </div>
                <div class="metric-caption">Indexed passages</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric3:
        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">RETRIEVAL</div>
                <div class="metric-value">FAISS</div>
                <div class="metric-caption">Vector similarity search</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    with metric4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">AI EXECUTION</div>
                <div class="metric-value">{DEVICE.upper()}</div>
                <div class="metric-caption">Current compute device</div>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        f"""
        <div class="success-banner">
            <strong>✓ Document ready</strong><br>
            {html.escape(st.session_state.document_name or "PDF")}
            has been processed. You can now ask questions about
            its extracted text.
        </div>
        """,
        unsafe_allow_html=True
    )


    # ========================================================
    # 15. QUESTION INPUT
    # ========================================================

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="section-label">Ask your document</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-heading">What would you like to know?</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-description">'
        'Ask a specific question and explore the passages '
        'that support the response.'
        '</div>',
        unsafe_allow_html=True
    )

    with st.form(
        "question_form",
        clear_on_submit=True
    ):

        question = st.text_input(
            "Your question",
            placeholder=(
                "e.g. What are the main findings of this document?"
            ),
            label_visibility="collapsed"
        )

        submitted = st.form_submit_button(
            "✦  Generate answer",
            type="primary",
            use_container_width=True
        )

    if submitted:

        if not question.strip():
            st.warning(
                "Please enter a question before generating an answer."
            )

        else:
            try:
                with st.spinner(
                    "Searching the document and generating your answer..."
                ):

                    embedding_model, tokenizer, llm = load_models()

                    results = retrieve_chunks(
                        question.strip(),
                        st.session_state.chunks,
                        st.session_state.index,
                        embedding_model
                    )

                    answer, source_pages = generate_answer(
                        question.strip(),
                        results,
                        tokenizer,
                        llm
                    )

                st.session_state.chat_history.append({
                    "question": question.strip(),
                    "answer": answer,
                    "source_pages": source_pages,
                    "results": results
                })

            except Exception as error:
                st.error(
                    f"Unable to generate an answer: {error}"
                )


    # ========================================================
    # 16. CONVERSATION HISTORY
    # ========================================================

    if st.session_state.chat_history:

        st.markdown("<br>", unsafe_allow_html=True)

        history_col, clear_col = st.columns(
            [4, 1]
        )

        with history_col:
            st.markdown(
                '<div class="section-heading">Conversation</div>',
                unsafe_allow_html=True
            )

            st.markdown(
                '<div class="section-description">'
                'Review your questions, answers, and supporting passages.'
                '</div>',
                unsafe_allow_html=True
            )

        with clear_col:
            if st.button(
                "Clear chat",
                use_container_width=True
            ):
                st.session_state.chat_history = []
                st.rerun()

        for i, item in enumerate(
            reversed(st.session_state.chat_history)
        ):

            with st.container(border=True):

                st.markdown(
                    '<div class="answer-label">YOUR QUESTION</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    f'<div class="question-card">'
                    f'{html.escape(item["question"])}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="answer-label">AI RESPONSE</div>',
                    unsafe_allow_html=True
                )

                st.markdown(
                    '<div class="assistant-card">'
                    f'{html.escape(item["answer"])}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                if item["source_pages"]:

                    page_labels = ", ".join(
                        f"Page {page}"
                        for page in item["source_pages"]
                    )

                    st.caption(
                        f"📄 Retrieved source pages: {page_labels}"
                    )

                with st.expander(
                    "Inspect retrieved passages and similarity scores"
                ):

                    if not item["results"]:
                        st.info(
                            "No passages were retrieved."
                        )

                    for source in item["results"]:

                        st.markdown(
                            f"""
                            <div class="source-card">
                                <div class="source-badge">
                                    PAGE {source["page"]}
                                </div>
                                <br>
                                {html.escape(source["text"])}
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                        st.progress(
                            min(
                                max(
                                    (source["score"] + 1) / 2,
                                    0.0
                                ),
                                1.0
                            ),
                            text=(
                                "Similarity score: "
                                f'{source["score"]:.3f}'
                            )
                        )


# ============================================================
# 17. EMPTY STATE
# ============================================================

else:

    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        """
        <div class="panel">
            <div style="font-size: 42px; margin-bottom: 12px;">
                📄
            </div>

            <div class="section-heading">
                Your knowledge workspace starts here
            </div>

            <div class="section-description">
                Upload a PDF using the sidebar to begin.
                DocuMind AI will extract the text, build a vector index,
                and prepare the document for semantic question answering.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="section-label">Your workflow</div>',
        unsafe_allow_html=True
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
            <div class="panel">
                <div style="font-size: 27px;">📄</div>
                <h3 style="font-size: 17px;">01. Upload</h3>
                <p style="font-size: 13px; color: #68738A;">
                    Add a readable PDF document to your workspace.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div class="panel">
                <div style="font-size: 27px;">🧠</div>
                <h3 style="font-size: 17px;">02. Understand</h3>
                <p style="font-size: 13px; color: #68738A;">
                    Convert document passages into semantic embeddings
                    and index them with FAISS.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col3:
        st.markdown(
            """
            <div class="panel">
                <div style="font-size: 27px;">💬</div>
                <h3 style="font-size: 17px;">03. Ask</h3>
                <p style="font-size: 13px; color: #68738A;">
                    Ask questions and inspect retrieved passages
                    with their source page numbers.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# 18. FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        DocuMind AI · Built with Python, Streamlit, FAISS
        and open-source transformer models.<br>
        AI-generated answers can be inaccurate. Verify important
        information against the original document.
    </div>
    """,
    unsafe_allow_html=True
)
