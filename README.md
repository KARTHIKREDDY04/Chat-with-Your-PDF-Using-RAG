%%writefile README.md

# 📚 Chat with Your PDF Using RAG

An AI-powered PDF question-answering application built using
Retrieval-Augmented Generation (RAG).

## 🚀 Features

- Upload PDF documents.
- Extract text from PDF pages.
- Split text into smaller chunks.
- Generate embeddings using Sentence Transformers.
- Retrieve relevant chunks using FAISS.
- Generate answers using FLAN-T5.
- Display source page numbers.
- Interactive web interface using Streamlit.

## 🧠 Technologies Used

- Python
- Streamlit
- PyPDF
- Sentence Transformers
- FAISS
- Hugging Face Transformers
- FLAN-T5

## 🏗️ Architecture

PDF Upload
    ↓
Text Extraction
    ↓
Text Chunking
    ↓
Embedding Generation
    ↓
FAISS Vector Search
    ↓
Relevant Context Retrieval
    ↓
FLAN-T5 Answer Generation
    ↓
Answer with Source Pages

## ⚙️ Installation

Clone the repository:

```bash
git clone YOUR_REPOSITORY_URL
