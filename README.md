# 📚 Chat with Your PDF Using RAG

### An AI-Powered PDF Question-Answering System Using Retrieval-Augmented Generation

**Python · Streamlit · FAISS · Sentence Transformers · Hugging Face Transformers**

---

## 📌 Table of Contents

1. [Project Overview](#-1-project-overview)
2. [Problem Statement](#-2-problem-statement)
3. [Project Objectives](#-3-project-objectives)
4. [What Is RAG?](#-4-what-is-retrieval-augmented-generation-rag)
5. [How the Project Works](#-5-how-the-project-works)
6. [System Architecture](#-6-system-architecture)
7. [Technologies Used](#-7-technologies-used)
8. [Project Folder Structure](#-8-project-folder-structure)
9. [Installation and Setup](#-9-installation-and-setup)
10. [Running the Application](#-10-running-the-application)
11. [Detailed Implementation](#-11-detailed-implementation)
12. [Understanding Embeddings](#-12-understanding-embeddings)
13. [Understanding FAISS](#-13-understanding-faiss)
14. [Understanding the Language Model](#-14-understanding-the-language-model)
15. [Example Usage](#-15-example-usage)
16. [Advantages of This Project](#-16-advantages-of-this-project)
17. [Limitations](#-17-limitations)
18. [Troubleshooting](#-18-troubleshooting)
19. [Future Improvements](#-19-future-improvements)
20. [Learning Outcomes](#-20-learning-outcomes)
21. [Frequently Asked Questions](#-21-frequently-asked-questions)
22. [Contributing](#-22-contributing)
23. [License](#-23-license)
24. [Author](#-24-author)

---

# 🚀 1. Project Overview

**Chat with Your PDF Using RAG** is an AI-powered document question-answering application that allows users to upload PDF documents and ask questions about their contents using natural language.

Instead of manually reading an entire document to find specific information, users can ask questions and receive answers generated from relevant sections of the uploaded PDF.

The project implements a technique called **Retrieval-Augmented Generation (RAG)**.

RAG combines two important capabilities:

- **Information retrieval:** Finds the sections of a document most relevant to a user's question.
- **Answer generation:** Uses a language model to generate a response based on the retrieved information.

The application uses Sentence Transformers to convert document text into numerical embeddings, FAISS to retrieve relevant text chunks, and FLAN-T5 to generate answers.

A Streamlit interface provides a simple way to interact with the system.

The application is designed to work without requiring a paid AI API.

## ✨ Key Features

- 📄 Upload PDF documents through a web interface.
- 📝 Extract text from PDF pages.
- ✂️ Divide documents into smaller text chunks.
- 🧠 Generate semantic embeddings using Sentence Transformers.
- 🔎 Search for relevant text using FAISS.
- 🤖 Generate answers using the FLAN-T5 language model.
- 📚 Display source page numbers for retrieved information.
- 💬 Ask questions in natural language.
- 🌐 Interact with the system through Streamlit.
- 💰 Run the core AI pipeline without a paid language-model API.
- 🔧 Extend the application with additional RAG features.

## 🎯 Intended Use Cases

This project can be adapted for:

- Research paper analysis.
- Academic study and revision.
- Textbook question answering.
- Technical documentation search.
- Business report analysis.
- User manuals and product documentation.
- Personal knowledge management.
- Learning and experimenting with AI applications.

The quality of the answers depends on the PDF's text quality, the retrieved context, and the language model's capabilities.

---

# 🧩 2. Problem Statement

PDF documents are widely used to store information, including research papers, textbooks, reports, manuals, and business documents.

However, finding specific information in lengthy documents can be time-consuming.

Traditional PDF readers allow users to search for words, but they do not necessarily understand the meaning behind a question.

For example, a user might ask:

> What are the main advantages of renewable energy?

A traditional keyword search may look for the exact words in the question. However, a document might discuss the same topic using different terminology, such as clean energy, sustainable power, or reduced carbon emissions.

A semantic retrieval system can help identify relevant passages even when the wording differs.

This project addresses the problem by combining semantic search with language-model-based answer generation.

### Proposed Solution

The application follows these steps:

1. Accept a PDF document from the user.
2. Extract readable text from the document.
3. Divide the text into manageable chunks.
4. Convert each chunk into an embedding.
5. Store the embeddings in a FAISS index.
6. Convert the user's question into an embedding.
7. Retrieve the most relevant chunks.
8. Provide those chunks to a language model.
9. Generate an answer based on the retrieved context.
10. Display the answer and associated source pages.

This approach helps users interact with documents more naturally.

---

# 🎯 3. Project Objectives

The primary objective is to build an end-to-end RAG application using open-source tools and locally loaded models.

### Technical Objectives

1. Understand the fundamental concepts of Retrieval-Augmented Generation.
2. Extract text from PDF documents using Python.
3. Implement text chunking with overlap.
4. Generate semantic embeddings using Sentence Transformers.
5. Create a vector index using FAISS.
6. Retrieve relevant document chunks using similarity search.
7. Generate contextual answers using FLAN-T5.
8. Build an interactive web application using Streamlit.
9. Display source page numbers to help users verify answers.
10. Organize the implementation into reusable Python functions.

### Learning Objectives

By completing this project, developers can learn how several AI components work together in a real application.

The project provides practical experience with:

- Natural language processing.
- Semantic search.
- Vector representations.
- Vector databases and similarity search.
- Transformer-based language models.
- Prompt construction.
- Document processing.
- AI application development.
- Model deployment considerations.

---

# 🧠 4. What Is Retrieval-Augmented Generation (RAG)?

Retrieval-Augmented Generation, commonly called RAG, is an approach that combines information retrieval with language generation.

A language model normally generates responses based on information learned during training and the context supplied in its prompt.

However, a model might not know the contents of a user's private PDF. Its training data may also be incomplete or outdated.

RAG helps address this limitation by retrieving relevant information from an external knowledge source before generating an answer.

In this project, the external knowledge source is the PDF uploaded by the user.

## The Two Main Components of RAG

### A. Retrieval

The retrieval component searches for information relevant to the user's question.

In this project:

- Sentence Transformers generates embeddings.
- FAISS searches the embedding index.
- The system selects the most relevant document chunks.

### B. Generation

The generation component uses a language model to produce an answer.

In this project:

- The retrieved chunks become the context.
- The question and context are combined into a prompt.
- FLAN-T5 generates the response.

## Why Use RAG Instead of Sending the Entire PDF to a Language Model?

Sending an entire large document to a language model can be inefficient or impossible when the document exceeds the model's input limit.

RAG instead retrieves a smaller selection of relevant passages.

Potential benefits include:

- Less irrelevant context.
- More focused answers.
- Better use of limited model input space.
- Easier source tracking.
- The ability to work with documents not included in the model's training data.

RAG does not guarantee that every answer will be correct. The retrieval component can select irrelevant passages, and the language model can still misunderstand or misrepresent the context.

---

# ⚙️ 5. How the Project Works

The application contains two major workflows:

1. Document ingestion and indexing.
2. Question answering.

## Workflow A: Document Ingestion

### Step 1: PDF Upload

The user uploads a PDF through the Streamlit interface.

The application receives the uploaded file and passes it to the PDF processing component.

### Step 2: Text Extraction

The application uses PyPDF to read the PDF and extract text from its pages.

Each extracted page is associated with its page number.

For example:

```text
Page 1:
Introduction to Artificial Intelligence...

Page 2:
Machine learning is a subset of artificial intelligence...

Page 3:
Deep learning uses neural networks...
```

Page numbers are retained so that the application can display source references later.

**Important:** PyPDF extracts existing text. Scanned documents that contain only images may require OCR, which is not implemented in the current version.

### Step 3: Text Chunking

Long documents are divided into smaller text segments called chunks.

The current implementation uses:

- Chunk size: 500 characters.
- Chunk overlap: 50 characters.

For example:

```text
Original document:
Artificial intelligence is transforming many industries...

Chunk 1:
Artificial intelligence is transforming many industries...

Chunk 2:
...many industries by automating repetitive tasks...
```

Overlapping chunks help preserve context near chunk boundaries.

However, splitting by character count can separate sentences or paragraphs. More advanced systems may use sentence-aware or token-aware chunking.

### Step 4: Embedding Generation

Each text chunk is converted into a numerical vector using:

`sentence-transformers/all-MiniLM-L6-v2`

The vector represents semantic information about the text.

Text passages with related meanings may have similar vectors, although similarity is not guaranteed to reflect every aspect of meaning.

### Step 5: Vector Index Creation

The generated embeddings are stored in a FAISS index.

The application uses `IndexFlatIP`, which performs an exact inner-product search.

Because the vectors are normalized, their inner product corresponds to cosine similarity.

The index makes it possible to retrieve relevant text chunks efficiently without comparing the question against every chunk manually in Python.

## Workflow B: Question Answering

### Step 6: User Asks a Question

The user enters a natural-language question.

Example:

```text
What is machine learning?
```

### Step 7: Question Embedding

The same embedding model converts the question into a numerical vector.

Using the same embedding model for the document chunks and questions keeps their vectors in the same embedding space.

### Step 8: Similarity Search

FAISS compares the question embedding with the stored chunk embeddings.

The current application retrieves up to three chunks.

These chunks are ranked by similarity score.

### Step 9: Context Construction

The retrieved text chunks are combined into a context string.

The context includes page labels to preserve source information.

### Step 10: Answer Generation

The context and question are passed to the FLAN-T5 model.

The prompt instructs the model to answer using the supplied context and to say when the answer cannot be found.

### Step 11: Answer Display

The application displays:

- The generated answer.
- The source page numbers associated with the retrieved chunks.
- The retrieved text in an expandable section.

The source pages indicate which pages supplied the context. They do not independently prove that every claim in the answer is supported by those pages.

---

# 🏗️ 6. System Architecture

The project follows a straightforward RAG architecture.

## High-Level Architecture

```text
                  USER
                    |
                    v
          +-------------------+
          | Streamlit Web App |
          +-------------------+
                    |
                    v
          +-------------------+
          |    PDF Upload     |
          +-------------------+
                    |
                    v
          +-------------------+
          | PyPDF Text        |
          | Extraction        |
          +-------------------+
                    |
                    v
          +-------------------+
          | Text Chunking     |
          | 500 Characters    |
          | 50 Overlap        |
          +-------------------+
                    |
                    v
          +-------------------+
          | Sentence          |
          | Transformers      |
          | Embeddings        |
          +-------------------+
                    |
                    v
          +-------------------+
          | FAISS Vector      |
          | Index             |
          +-------------------+
                    |
                    |
          USER QUESTION
                    |
                    v
          +-------------------+
          | Question          |
          | Embedding         |
          +-------------------+
                    |
                    v
          +-------------------+
          | FAISS Similarity  |
          | Search            |
          +-------------------+
                    |
                    v
          +-------------------+
          | Relevant Chunks   |
          | and Page Numbers  |
          +-------------------+
                    |
                    v
          +-------------------+
          | Prompt Construction|
          +-------------------+
                    |
                    v
          +-------------------+
          | FLAN-T5 Language  |
          | Model              |
          +-------------------+
                    |
                    v
          +-------------------+
          | Answer and Source |
          | Pages             |
          +-------------------+
                    |
                    v
                   USER
```

## Component Responsibilities

| Component | Responsibility |
|---|---|
| Streamlit | Provides the user interface |
| PyPDF | Extracts text from PDF pages |
| Text chunking | Divides text into smaller segments |
| Sentence Transformers | Converts text into embeddings |
| FAISS | Searches for relevant embeddings |
| FLAN-T5 | Generates a contextual answer |
| PyTorch | Runs the language model |
| NumPy | Handles numerical embedding arrays |

## Data Flow

The project transforms information through several stages:

**PDF → Extracted Text → Chunks → Embeddings → FAISS Index**

When the user asks a question:

**Question → Question Embedding → Similarity Search → Retrieved Context → Language Model → Answer**

This separation between indexing and answering is a fundamental RAG design pattern.

---

# 🛠️ 7. Technologies Used

## 7.1 Python

Python is the primary programming language used to implement the application.

It provides libraries for document processing, numerical computing, machine learning, and web application development.

## 7.2 Streamlit

Streamlit creates the interactive web interface.

It allows users to upload PDFs, enter questions, view answers, and inspect retrieved passages without building a frontend from scratch.

Official documentation: https://docs.streamlit.io/

## 7.3 PyPDF

PyPDF extracts text from PDF pages and provides access to page-level information.

Official documentation: https://pypdf.readthedocs.io/

## 7.4 Sentence Transformers

Sentence Transformers converts text into dense numerical vectors known as embeddings.

This project uses:

`sentence-transformers/all-MiniLM-L6-v2`

The model produces 384-dimensional embeddings.

Official documentation: https://sbert.net/

## 7.5 FAISS

FAISS is a library for efficient similarity search and clustering of dense vectors.

This project uses an exact inner-product index with normalized embeddings.

Official repository: https://github.com/facebookresearch/faiss

## 7.6 Hugging Face Transformers

Transformers provides the tokenizer and model-loading functionality used for FLAN-T5.

Official documentation: https://huggingface.co/docs/transformers/

## 7.7 FLAN-T5

The project uses:

`google/flan-t5-base`

FLAN-T5 is an instruction-tuned text-to-text transformer model.

It receives the question and retrieved context and generates a text response.

Model page: https://huggingface.co/google/flan-t5-base

## 7.8 PyTorch

PyTorch runs the language model and provides the inference operations used during answer generation.

The application uses a GPU when available and otherwise falls back to the CPU.

Official documentation: https://pytorch.org/docs/stable/index.html

## 7.9 NumPy

NumPy handles numerical arrays.

The project converts embeddings to 32-bit floating-point arrays before adding them to FAISS.

Official documentation: https://numpy.org/doc/

---

# 📁 8. Project Folder Structure

The repository has the following structure:

```text
chat-with-your-pdf-rag/
│
├── app.py
│
├── requirements.txt
│
├── README.md
│
├── .gitignore
│
└── .streamlit/
    └── config.toml       # Optional configuration
```

### `app.py`

Contains the Streamlit interface and RAG implementation.

It includes functions for:

- Loading models.
- Extracting PDF text.
- Creating text chunks.
- Generating embeddings.
- Creating the FAISS index.
- Retrieving relevant chunks.
- Generating answers.
- Displaying source page numbers.

### `requirements.txt`

Lists the Python dependencies required to run the application.

### `README.md`

Documents the project, architecture, installation, usage, and implementation.

### `.gitignore`

Prevents selected files, such as private PDFs and environment files, from being tracked by Git.

### `.streamlit/config.toml`

This optional configuration file can be used to customize Streamlit settings.

It is not required for the current application.

---

# 💻 9. Installation and Setup

## Prerequisites

Before running the application, ensure that you have:

- Python installed.
- pip available.
- An internet connection for initial package and model downloads.
- Sufficient RAM for the embedding and language models.
- A compatible environment for PyTorch and FAISS.

A GPU is helpful but not mandatory.

The exact resource requirements depend on the document size, installed package versions, and available hardware.

## Step 1: Clone the Repository

Replace `YOUR_USERNAME` with your GitHub username.

```bash
git clone https://github.com/YOUR_USERNAME/chat-with-your-pdf-rag.git
```

## Step 2: Enter the Project Directory

```bash
cd chat-with-your-pdf-rag
```

## Step 3: Create a Virtual Environment

On Windows:

```bash
python -m venv .venv
```

Activate it:

```bash
.venv\Scripts\activate
```

On Linux or macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

A virtual environment isolates the project's Python packages from other projects.

## Step 4: Upgrade pip

```bash
python -m pip install --upgrade pip
```

## Step 5: Install Dependencies

```bash
pip install -r requirements.txt
```

This installs the libraries required by the application.

Some machine-learning dependencies may require additional system-specific configuration. If installation fails, check the error message and the official package documentation.

---

# ▶️ 10. Running the Application

After installing the dependencies, start Streamlit:

```bash
streamlit run app.py
```

Streamlit will display a local address in the terminal, commonly:

`http://localhost:8501`

Open the displayed address in your browser.

## Using the Application

### Step 1: Upload a PDF

Click the upload control and select a readable PDF document.

### Step 2: Wait for Processing

The application extracts the document text, divides it into chunks, generates embeddings, and creates the FAISS index.

The first run may take longer because the models must be downloaded and loaded.

### Step 3: Enter a Question

Type a question related to the document.

For example:

```text
What are the main findings of this report?
```

### Step 4: Generate the Answer

Click **Get Answer**.

The system retrieves relevant text chunks and passes them to FLAN-T5.

### Step 5: Inspect the Sources

Review the source page numbers and expand the retrieved-text section to inspect the passages used to construct the answer.

### Step 6: Ask Another Question

Enter another question about the same document.

In the current implementation, the PDF is processed again when Streamlit reruns the application. Caching the document index would improve efficiency in a future version.

---

# 🔬 11. Detailed Implementation

This section explains the major functions in the application.

## 11.1 Loading the Models

The application loads an embedding model and a text-generation model.

The embedding model is responsible for semantic representations, while FLAN-T5 generates answers.

Streamlit's `st.cache_resource` decorator helps avoid reloading these models on every rerun within the same application process.

## 11.2 Extracting PDF Text

The application uses PyPDF to iterate through the pages.

For each page, it:

1. Extracts readable text.
2. Assigns the page number.
3. Divides the text into overlapping chunks.
4. Stores each chunk with its page number.

The stored structure resembles:

```python
{
    "text": "Machine learning is a field of artificial intelligence...",
    "page": 3
}
```

Keeping the page number alongside the text enables source-page display later.

## 11.3 Creating Embeddings

The application sends the text chunks to the Sentence Transformers model.

The model returns a numerical vector for each chunk.

The vectors are normalized before indexing.

Normalization allows the inner product between two vectors to represent cosine similarity.

## 11.4 Creating the FAISS Index

The application initializes:

```python
faiss.IndexFlatIP(dimension)
```

The index stores the chunk embeddings and searches them using inner product.

Because the vectors are normalized, higher similarity scores indicate closer alignment in the embedding space.

The index performs exact search rather than approximate nearest-neighbor search.

## 11.5 Retrieving Relevant Chunks

When a user submits a question, the application:

1. Generates the question embedding.
2. Searches the FAISS index.
3. Selects up to three matching chunks.
4. Collects their text and page numbers.
5. Returns the results for answer generation.

The number of retrieved chunks is controlled by `top_k`.

A larger value can provide more context but may also introduce irrelevant information or exceed the model's useful input capacity.

## 11.6 Constructing the Prompt

The application combines the retrieved text with the user's question.

The prompt instructs FLAN-T5 to use the provided context and acknowledge when the answer is unavailable.

This encourages grounded answers, but prompt instructions alone cannot guarantee that the model will never hallucinate.

## 11.7 Generating the Answer

The tokenizer converts the prompt into model inputs.

The model generates a response using deterministic decoding settings.

The tokenizer then converts the generated tokens into readable text.

The final response is displayed in Streamlit alongside the retrieved source pages.

---

# 🧮 12. Understanding Embeddings

An embedding is a numerical representation of text.

For example, consider these sentences:

1. "A dog is playing in the garden."
2. "A puppy is running outside."
3. "The stock market declined yesterday."

The first two sentences have related meanings, even though they use different words.

An embedding model attempts to represent their meanings in a numerical space.

The first two vectors may therefore be closer together than either is to the third vector.

## Why Are Embeddings Important in This Project?

Keyword search depends heavily on matching words.

Semantic retrieval instead compares numerical representations of text and questions.

This helps the application retrieve passages that may be relevant even when they do not contain exactly the same words as the user's question.

## Embedding Dimensions

The model used in this project produces a 384-dimensional vector for each text input.

Conceptually, one embedding might look like:

```text
[0.12, -0.08, 0.31, ..., -0.04]
```

These numbers are learned representations, not individually interpretable measurements.

The full vector captures patterns that the model has learned from text.

---

# 🔎 13. Understanding FAISS

FAISS stands for **Facebook AI Similarity Search**.

It is a library for searching and clustering dense vectors.

In this project, FAISS is used to find the document chunks whose embeddings are most similar to a question embedding.

## How Similarity Search Works

Suppose the document contains these chunks:

| Chunk | Content |
|---|---|
| Chunk 1 | Introduction to artificial intelligence |
| Chunk 2 | Explanation of machine learning |
| Chunk 3 | Information about cooking |
| Chunk 4 | Applications of deep learning |

If the question is:

"What is machine learning?"

The retrieval system should ideally rank Chunk 2 highly and may also retrieve related material from Chunks 1 and 4.

Chunk 3 would generally be less relevant.

The exact ranking depends on the embeddings and similarity scores.

## Why Use FAISS?

FAISS provides:

- Vector indexing.
- Similarity search.
- Efficient operations on numerical vectors.
- Multiple indexing strategies.
- CPU-based search, with additional GPU options available in supported configurations.

## Current Index Type

The project uses `IndexFlatIP`.

This index:

- Stores the vectors directly.
- Uses inner product as the search measure.
- Performs exact search.
- Does not require training an approximate index.

For a small PDF, this is a straightforward choice.

For very large collections, approximate indexes or other vector-search systems may offer better performance.

---

# 🤖 14. Understanding the Language Model

The application uses the `google/flan-t5-base` model.

FLAN-T5 is a text-to-text transformer model trained to follow instructions across a variety of tasks.

It can generate text in response to a prompt containing a question and supporting context.

## What Does the Language Model Receive?

The model receives a prompt containing:

- Instructions about how to answer.
- The retrieved document chunks.
- The user's question.
- A location where the answer should begin.

Conceptually:

```text
Instruction:
Answer using only the supplied context.

Context:
Machine learning is a branch of artificial intelligence...

Question:
What is machine learning?

Answer:
```

The model then generates a response.

## Why FLAN-T5?

It can be loaded through the Hugging Face Transformers library and run locally when sufficient resources are available.

This avoids the need for a paid hosted language-model API.

However, running locally does not mean that no internet connection is ever needed. The model files must normally be downloaded the first time unless they are already cached.

## CPU and GPU Execution

The application checks whether CUDA is available.

If a compatible GPU is available, it uses the GPU. Otherwise, it uses the CPU.

CPU inference can be slower, particularly when loading or running the larger language model.

---

# 🧪 15. Example Usage

Consider a hypothetical PDF about machine learning.

The document contains:

> Machine learning is a branch of artificial intelligence that enables systems to learn patterns from data and use those patterns to make predictions.

A user asks:

**Question:** What is machine learning?

A possible answer is:

**Answer:** Machine learning is a branch of artificial intelligence that allows systems to learn patterns from data and use them to make predictions.

**Source pages:** Page 2

This is an illustrative example. The actual answer depends on the PDF contents and the model's output.

## Example 2: Question About an Unavailable Topic

Suppose the PDF discusses machine learning but contains no information about ocean pollution.

The user asks:

"What are the major causes of ocean pollution?"

The application instructs the model to acknowledge when the information is not present.

However, the model may still produce an unsupported answer. The current implementation does not enforce a strict similarity threshold or independently verify each claim.

For stronger reliability, future versions should reject low-confidence retrieval results and evaluate whether the retrieved context actually supports the generated answer.

---

# 🌟 16. Advantages of This Project

## 16.1 Natural-Language Interaction

Users can ask questions rather than manually search through a document.

## 16.2 Semantic Retrieval

Embeddings help retrieve conceptually relevant passages instead of relying exclusively on exact keyword matches.

## 16.3 Source-Page Information

The application retains page numbers so users can return to the original document and verify information.

## 16.4 No Paid AI API Required

The current implementation uses locally loaded models rather than requiring a paid inference API.

Hardware, storage, electricity, and hosting may still incur costs.

## 16.5 Modular Design

Document processing, embedding generation, retrieval, and answer generation are implemented as separate functions.

This makes the project easier to understand and extend.

## 16.6 Portfolio Value

The project demonstrates practical knowledge of:

- Python programming.
- Natural language processing.
- Embedding models.
- Vector search.
- RAG architecture.
- Transformer inference.
- Web application development.

---

# ⚠️ 17. Limitations

Understanding the limitations is important when evaluating an AI application.

## 17.1 Hallucinations

The language model can generate an incorrect answer even when the prompt instructs it to use only the provided context.

## 17.2 Retrieval Errors

The embedding model may retrieve chunks that are related to the question but do not contain the information needed to answer it.

## 17.3 PDF Extraction Limitations

Scanned PDFs may contain images rather than extractable text.

The current implementation does not include OCR.

## 17.4 Chunking Limitations

Splitting text by character count can break sentences, tables, and logical sections.

## 17.5 Context Length

The prompt is truncated to fit the configured model input limit.

Some retrieved information may therefore be omitted from the final model input.

## 17.6 Performance

FLAN-T5 may run slowly on a CPU.

Large PDFs can also take time to process and embed.

## 17.7 Source Attribution

The application reports the pages associated with retrieved chunks.

It does not verify that every statement in the answer is supported by those pages.

## 17.8 Document Privacy

Uploaded files are processed by the application. Their handling depends on the environment where the app is running.

A publicly exposed tunnel or deployment can make the application accessible to other people.

Do not upload confidential documents to an untrusted public deployment.

## 17.9 Index Persistence

The current application rebuilds its index when the document is processed. It does not implement a persistent document database.

---

# 🛠️ 18. Troubleshooting

## Problem 1: ModuleNotFoundError

**Possible cause:** A required package is not installed.

**Solution:**

```bash
pip install -r requirements.txt
```

Make sure you are running the command in the same Python environment used to launch Streamlit.

## Problem 2: The Application Is Slow

**Possible causes:**

- The language model is running on the CPU.
- The models are downloading for the first time.
- The PDF contains many pages.
- Embeddings are being regenerated after a Streamlit rerun.

**Possible improvements:**

- Cache document indexes.
- Reuse embeddings.
- Use a supported GPU when available.
- Process large documents in batches.

## Problem 3: No Text Found in the PDF

**Possible cause:** The PDF may contain scanned images instead of extractable text.

**Solution:** Add an OCR pipeline, such as one based on Tesseract, in a future version.

## Problem 4: The Answer Is Incorrect

**Possible causes:**

- The relevant passage was not retrieved.
- The question is ambiguous.
- The prompt omitted important information.
- The language model misunderstood the context.

**Possible improvements:**

- Inspect the retrieved chunks.
- Adjust the number of retrieved chunks.
- Improve chunking.
- Add a retrieval similarity threshold.
- Evaluate the model on a test dataset.

## Problem 5: Streamlit Fails to Load Through a Tunnel

**Possible cause:** A tunnel or proxy may interfere with the loading of Streamlit's frontend resources.

**Solution:** Test the application locally first:

```bash
streamlit run app.py
```

If the local application works but the tunneled application fails, investigate the tunnel configuration rather than immediately rewriting the RAG pipeline.

---

# 🔮 19. Future Improvements

The current implementation provides a foundation that can be expanded into a more capable document intelligence system.

## 19.1 Conversation History

Store previous questions and answers so users can have a continuous conversation about the document.

## 19.2 Multiple PDF Support

Allow users to upload multiple PDFs and search across all of them.

## 19.3 Persistent Vector Storage

Save embeddings and document metadata so the application does not need to rebuild the entire index every time.

## 19.4 Better Chunking

Use sentence-aware, paragraph-aware, or token-aware chunking to preserve more meaningful context.

## 19.5 OCR Support

Add optical character recognition to process scanned PDFs and image-based documents.

## 19.6 Retrieval Evaluation

Create a test dataset containing questions, expected answers, and relevant document passages.

Evaluate how often the retrieval component returns useful evidence.

## 19.7 Similarity Threshold

Reject low-similarity retrieval results instead of always passing the nearest chunks to the language model.

The threshold should be calibrated using evaluation data.

## 19.8 Improved Source Citations

Display source excerpts with page numbers and evaluate whether the generated answer is supported by the cited passages.

## 19.9 Alternative Language Models

Compare FLAN-T5 with other locally runnable instruction-tuned models.

Evaluate answer quality, latency, memory requirements, and hardware compatibility.

## 19.10 User Authentication

Add authentication and access controls before deploying the application for multiple users or private documents.

## 19.11 Automated Testing

Add tests for PDF extraction, chunking, retrieval, and answer-generation behavior.

## 19.12 Deployment

Deploy the application on an appropriate hosting platform after evaluating resource limits, privacy requirements, and model compatibility.

---

# 📈 20. Learning Outcomes

This project provides hands-on experience with an end-to-end RAG workflow.

After completing it, a developer should be able to explain:

1. Why document chunking is useful.
2. What embeddings represent.
3. How semantic similarity differs from keyword matching.
4. How FAISS retrieves similar vectors.
5. Why retrieval happens before generation in RAG.
6. How retrieved text is included in a model prompt.
7. How a local language model generates an answer.
8. Why source-page information is useful.
9. Why RAG systems can still produce hallucinations.
10. How to build a basic AI-powered web application.

The project also provides a starting point for more advanced work in agentic AI, retrieval evaluation, and document intelligence.

---

# ❓ 21. Frequently Asked Questions

## Q1. What is RAG?

RAG stands for Retrieval-Augmented Generation. It retrieves relevant information from an external source and supplies that information to a language model to help generate an answer.

## Q2. Does this project require a paid API?

No paid AI API is required by the current implementation. The models run through local Python libraries.

An internet connection is generally needed to download packages and models initially.

## Q3. Why are embeddings necessary?

Embeddings represent text as numerical vectors. They allow the application to compare the semantic similarity between document chunks and user questions.

## Q4. Why is FAISS used?

FAISS stores the document embeddings in an index and retrieves chunks whose vectors are most similar to the question vector.

## Q5. Why does the project use FLAN-T5?

FLAN-T5 generates text responses based on the question and the retrieved context.

## Q6. Can this project handle scanned PDFs?

Not reliably in its current form. Scanned PDFs generally need OCR to convert images of text into machine-readable text.

## Q7. Can I use this project in Google Colab?

Yes. The individual components can run in Colab, subject to the available runtime resources.

For a persistent application, a suitable deployment environment is preferable to relying on a temporary notebook session.

## Q8. Can I deploy this project publicly?

Yes, provided you choose a compatible hosting environment and consider the application's model size, memory requirements, security, and privacy.

## Q9. Does the application always provide correct answers?

No. Retrieval and generation can both fail. Answers should be checked against the original document, particularly for important decisions.

## Q10. Can I extend this project into an advanced RAG system?

Yes. Useful improvements include hybrid search, reranking, persistent storage, multi-document retrieval, evaluation, conversational memory, and stronger source verification.

---

# 🤝 22. Contributing

Contributions and suggestions are welcome.

Potential contributions include:

- Improving PDF text extraction.
- Implementing OCR.
- Adding support for multiple PDFs.
- Improving retrieval quality.
- Adding automated tests.
- Improving source citations.
- Optimizing model performance.
- Improving the user interface.

### Suggested Contribution Workflow

1. Fork the repository.
2. Create a new branch.
3. Implement your changes.
4. Test the application.
5. Submit a pull request describing the improvements.

---

# 📄 23. License

This project does not currently specify a repository license.

Before adding a license, choose one that matches how you want others to use, modify, and distribute your code.

Also review the licenses and usage conditions of the models and libraries used by the project.

---

# 👨‍💻 24. Author

**Name:** Rokkam Karthikeya Reddy

**GitHub:** https://github.com/KARTHIKREDDY04/

**Project:** Chat with Your PDF Using RAG

### About This Project

This project demonstrates the implementation of a basic Retrieval-Augmented Generation system that combines semantic retrieval and language-model-based answer generation.

It is intended for educational purposes and as a foundation for further experimentation with document intelligence and AI application development.

---

## ⭐ If You Find This Project Useful

Consider starring the repository, sharing feedback, or contributing improvements.

**Built with Python, open-source AI tools, and a practical approach to Retrieval-Augmented Generation.**
