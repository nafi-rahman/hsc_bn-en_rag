HSC-Bangla RAG (Level-1 Assessment)
A multilingual Retrieval-Augmented Generation system that answers English & Bangla questions from the HSC26 Bangla 1st Paper PDF.

🚀 Quick Start (One-Command)
# Clone the repository
git clone [https://github.com/your-username/hsc-bn-en-rag.git](https://github.com/your-username/hsc-bn-en-rag.git)
cd hsc-bn-en-rag

# 1. Place your PDF in data/00_raw_pdf/raw.pdf
# 2. Run the one-liner command
docker compose up --build

Then, you can access the different services:

Chat UI → http://localhost:8501

FastAPI docs → http://localhost:8000/docs

Qdrant dashboard → http://localhost:6333

📦 Tools & Libraries
Layer

Tool / Library

PDF text extraction

PyMuPDF (via langchain.document_loaders)

Cleaning & chunking

chunk_plan_a.py (RecursiveCharacterTextSplitter)

Dense embedding

shihab17/bangla-sentence-transformer

Sparse embedding

Qdrant/bm42-all-minilm-l6-v2-attentions

Vector DB

Qdrant

LLM

Google Gemini 1.5 Flash

API

FastAPI

Frontend

Streamlit

Containerisation

Docker & docker-compose

📚 Setup Guide
Prerequisites

Docker & Docker Compose

A Google Gemini API key

One-time ingestion (This is optional; the container auto-detects and processes data on startup)

poetry run python data/scripts/chunk_plan_a.py
poetry run python data/scripts/embed_and_upsert_qdrant.py

Run

docker compose up --build

🔌 API Documentation
Method

Endpoint

Description

POST

/retrieve

Accepts a JSON payload {"query": "..."} and returns the top-k relevant chunks.

You can explore the interactive API documentation, powered by OpenAPI, at http://localhost:8000/docs.

🧪 Evaluation Matrix
Metric

Value

Tool

Exact-match accuracy

auto-computed in Streamlit

Regex

Groundedness

cosine(answer vs chunks)

sentence-transformers

Relevance

cosine(query vs chunks)

sentence-transformers

🗣️ Sample Queries & Outputs
Bangla
Q: অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? A: শম্ভুনাথ

English
Q: Who is the fate deity of Anupam? A: His maternal uncle (মামা).

❓ Assessment Answers
1. PDF text extraction method & challenges
Method: I used PyMuPDF, accessed via the langchain.document_loaders module, for its efficiency and accuracy in parsing PDF text and metadata.

Challenge: The primary challenge was handling non-prose elements like MCQ tables and page headers/footers that cluttered the extracted text. I addressed this by implementing a clean() function that uses regular expressions to strip these unwanted patterns before chunking.

2. Chunking strategy
Strategy: I opted for a RecursiveCharacterTextSplitter strategy with a chunk size of 512 tokens and an overlap of 30 tokens.

Why: This approach balances providing sufficient context within each chunk against the need for retrieval granularity. The 512-token size is large enough to encapsulate complete paragraphs, preserving semantic flow, while the small overlap helps maintain context between adjacent chunks without excessive data duplication.

3. Embedding choice
Dense Embedding: I chose shihab17/bangla-sentence-transformer, a 768-dimensional model specifically trained on Bengali text. This is crucial for understanding the semantic nuances of queries in Bangla.

Sparse Embedding: For lexical matching, I used Qdrant's recommended Qdrant/bm42-all-minilm-l6-v2-attentions model. This sparse vector model (based on SPLADE) effectively captures keyword signals and is robust in handling out-of-vocabulary (OOV) words, which is common with specific Bengali terms.

4. Similarity & storage
Similarity Search: The system employs a hybrid search methodology. It computes dense vector similarity using the cosine distance metric for semantic relevance and combines it with sparse vector scores from BM42 for keyword-based matching. This dual approach ensures both conceptually similar and lexically exact matches are found.

Storage: I used a single Qdrant collection configured with two named vectors: dense for the Bangla sentence transformer embeddings and bm42 for the sparse embeddings. This setup allows for efficient hybrid queries against both vector types simultaneously.

5. Handling vague queries
Fallback Mechanism: A simple yet effective threshold is implemented. If the maximum cosine similarity score of the retrieved chunks falls below 0.35, the system concludes that it lacks a confident context. In this case, it returns a predefined, honest response: “আমি নিশ্চিত নই” ("I am not sure").

Future Work: This could be enhanced by implementing query expansion techniques (e.g., using a synonym dictionary) or a re-ranking model to better discern user intent from ambiguous phrasing.

6. Relevance observations
Current Performance: The system achieves 3 out of 3 exact hits on the initial set of test cases, demonstrating high relevance for the targeted queries.

Potential Improvements: To further boost relevance, I would consider:

Expanding the Corpus: Including more related documents to enrich the knowledge base.

Fine-tuning Chunking: Experimenting with sentence-level chunking for highly factual Q&A.

Adding a Reranker: Implementing a cross-encoder model to re-rank the top-k retrieved results for finer-grained relevance scoring before generation.

📁 Repository Layout
.
├── rag/                    # Core retrieval and generator logic
│   ├── __init__.py
│   ├── generator.py        # LLM interaction module
│   └── retriever.py        # Qdrant hybrid search module
├── data/
│   ├── 00_raw_pdf/
│   │   └── raw.pdf         # Input PDF goes here
│   └── scripts/            # Data ingestion pipeline
│       ├── __init__.py
│       ├── chunk_plan_a.py
│       └── embed_and_upsert_qdrant.py
├── app.py                  # Streamlit frontend application
├── api.py                  # FastAPI backend
├── Dockerfile
├── docker-compose.yml
└── README.md

📄 License
This project is licensed under the MIT License.
