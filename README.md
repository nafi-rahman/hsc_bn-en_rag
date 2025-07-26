# 🧠 Multilingual RAG System (Bangla + English)

A lightweight, multilingual Retrieval-Augmented Generation (RAG) system designed to answer English and Bengali queries from a given Bangla book (`HSC26 Bangla 1st Paper`). It retrieves semantically relevant chunks from a preprocessed document corpus and generates grounded responses using a language model.

---

## 🚀 Project Overview

This system:

- Accepts **Bangla** questions (and English, but the UI prompt & system prompt are tuned for **Bangla**).
- Retrieves relevant **512-token paragraph chunks** from the HSC-Bangla PDF corpus stored in **Qdrant** (`hsc26_bangla_hybrid`).
- Generates **concise answers** grounded in the retrieved context using **Gemini-1.5-Flash** (`rag.generator.generate_answer`).
- Supports **short-term memory** (`st.session_state.history`) inside the Streamlit chat and **long-term memory** via a **dense + sparse hybrid semantic search** (`retrieval.py`).
- Exposes a **lightweight REST API** (`/retrieve` on `http://localhost:8000`) backed by **FastAPI** for external retrieval calls.

---

## 📁 Folder Structure

```
HSC_BN-EN_RAG/
├── data/
│   ├── 00_raw_pdf/
│   │   └── raw.pdf                       # Original PDF (text-based)
│   └── processed/
│       ├── chunks_a_output.jsonl        # Chunked content (JSONL format)
│       └── hybrid_output.jsonl          # Final processed output
│
├── scripts/
│   ├── chunk_plan_a.py                  # Chunking strategy
│   ├── embed_and_upsert_qdrant.py       # Embedding + Qdrant upload
│   └── gemini_raw_extract.py            # Gemini-based extraction
│
├── rag/
│   ├── __pycache__/                     # Compiled Python cache
│   ├── generator.py                     # Likely handles prompt generation
│   ├── models.py                        # Data models / pydantic / schema
│   └── retrieval.py                     # Retrieval logic (hybrid, dense, etc.)
│
├── app.py                               # Main Streamlit app
│
├── start_app.ps1/start_app.bash         #script to create venv and pip install(windows                                           or linux)
├── requirements.txt                     #libraries to install
├── poetry.lock                          # Poetry lockfile for reproducible deps
├── pyproject.toml                       # Poetry config + project deps
├── .python-version                      # Python version (likely for `pyenv`)
└── README.md                            # Project overview / instructions

```

---



## ⚙️ Setup Guide

### 🔧 Installation

```bash
# 1. Clone the repo
git clone <https://github.com/yourusername/multilingual-rag.git>
cd multilingual-rag

# 2. Pin Python 3.11.x (pyenv recommended but not mandatory)
pyenv install 3.11.8
pyenv local 3.11.8

```

### 🐳 Start Qdrant (Docker required)

**Windows PowerShell**

```powershell
docker run -d --name qdrant `
  -p 6333:6333 -p 6334:6334 `
  -v ${PWD}/qdrant_storage:/qdrant/storage `
  qdrant/qdrant

```

**Linux / macOS**

```bash
docker run -d --name qdrant \\
  -p 6333:6333 -p 6334:6334 \\
  -v $(pwd)/qdrant_storage:/qdrant/storage \\
  qdrant/qdrant

```

> Wait until docker logs qdrant shows "Qdrant HTTP server is running".
> 

### 🚀 One-command bootstrap

| OS | Command |
| --- | --- |
| **Windows** | `powershell -ExecutionPolicy Bypass -File start_app.ps1` |
| **Linux/macOS** | `chmod +x start_app.bash && ./start_app.bash` |

### ▶️ how to run the app

```bash
# 1. Install deps & activate venv
source .venv/bin/activate   # or .\\.venv\\Scripts\\activate

# 2. Start FastAPI retrieval service
uvicorn rag.retrieval:app --host 0.0.0.0 --port 8000

# 3. In another shell, launch the chat UI
streamlit run app.py --server.port 8501

```

Visit

- **Chat UI**: [http://localhost:8501](http://localhost:8501/)
- **Retrieval API docs**: http://localhost:8000/docs

---

## 📚 Tools, Libraries & Models Used

| Purpose / Layer | Tool / Library / Model | Version Constraint | Notes |
| --- | --- | --- | --- |
| **Python Runtime** | `python` | ≥3.11, <3.14 | Locked via `.python-version` |
| **Project & Dependency Management** | `poetry` | – | Declared in `pyproject.toml` |
| **Web Framework** | `fastapi` | 0.116.x | REST API (`/retrieve`) |
| **ASGI Server** | `uvicorn` | 0.35.x | Serves the FastAPI app |
| **Frontend / UI** | `streamlit` | 1.47.x | Chat & evaluation tabs |
| **LLM Provider** | `langchain-google-genai` | 2.1.x | Access to **Gemini-1.5-Flash** |
| **LLM Model** | **gemini-1.5-flash** | – | Concise Bangla answer generation |
| **Vector DB** | `qdrant-client` | 1.15.x | Storage & retrieval (`hsc26_bangla_hybrid`) |
| **Dense Embedding** | `sentence-transformers` | 5.x | Loads **shihab17/bangla-sentence-transformer** |
| **Dense Model** | **shihab17/bangla-sentence-transformer** | – | 768-dim cosine vectors |
| **Sparse Embedding** | `fastembed` | 0.7.x | BM42 sparse vectors |
| **Sparse Model** | **Qdrant/bm42-all-minilm-l6-v2-attentions** | – | Sparse vector generation |
| **Re-ranker** | `sentence-transformers` (via `CrossEncoder`) | 5.x | Loads **cross-encoder/mmarco-mMiniLMv2-L12-H384-v1** |
| **Re-rank Model** | **cross-encoder/mmarco-mMiniLMv2-L12-H384-v1** | – | Bangla-friendly cross-encoder |
| **LangChain Abstractions** | `langchain`, `langchain-qdrant`, `langchain-huggingface` | 0.3.x | Chains, vector-store wrappers |
| **Data Validation** | `pydantic` | 2.11.x | `RetrievedChunk`, `Answer`, FastAPI request/response models |
| **Chunking & Text Processing** | `langchain.text_splitter` | via `langchain` | Recursive splitter (512-tok, 30-tok overlap) |
| **Evaluation Metrics** | `nltk`, `rouge-score`, `scikit-learn` | latest compatible | BLEU-1, ROUGE-1, ROUGE-L, Token-F1 |
| **Environment Management** | `pyenv` + `.python-version` | – | Ensures correct Python version |
| **Installation Scripts** | `start_app.ps1` / `start_app.bash` | – | One-command venv + dependencies |

---

## 🧼 Preprocessing

- Upstream extraction (`gemini_raw_extract.py`) delivers page-level text/tables to `hybrid_output.jsonl`; all downstream steps operate on this already-cleaned JSONL.
- **Unicode normalization** (`unicodedata.normalize("NFKC")`).
- **Strip markdown fences, MCQ tables, page headers** (`chunk_plan_a.py::clean`).
- **Sentence segmentation** via `RecursiveCharacterTextSplitter` using Bangla-aware separators: `\n\n`, `\n`, `।` , `.` .
- **Low-content filtering** implicitly handled by the splitter—empty or whitespace-only chunks are skipped at upsert time.

---

## 🧹 Chunking Strategy

- **Type**: Recursive sentence-based merging up to a token limit.
- **Max Tokens**: 512 (via `tiktoken` proxy or character fallback).
- **Overlap**: 30 tokens to preserve context continuity.
- **Rationale**: Keeps semantic boundaries (Bangla punctuation) while avoiding truncation inside sentences.

---

## 🧠 Embedding Model

- **Model**: **shihab17/bangla-sentence-transformer** (loaded via `sentence-transformers`)
- **Why**: Trained specifically for Bangla sentence pairs, handles code-mixed English, and yields 768-dim cosine vectors proven to capture HSC-level semantics.
- **How**: Each 512-token chunk is encoded into a dense vector; these vectors, together with BM42 sparse vectors, are upserted into Qdrant so both semantic and lexical similarity drive retrieval.

---

## 🔍 Retrieval Logic

- **Hybrid search** via **Qdrant**– dense cosine similarity on `shihab17/bangla-sentence-transformer` vectors– sparse BM42 scores on `Qdrant/bm42-all-minilm-l6-v2-attentions` vectors
- **Top-k filtering** (`FETCH_K=30` per modality → fused to `K_AFTER_RRF=15` via Reciprocal Rank Fusion → final `K_FINAL=3`)
- **Re-ranking** with **cross-encoder/mmarco-mMiniLMv2-L12-H384-v1**– Bangla-optimized cross-encoder re-scores the top 15 candidates → returns the 3 best passages, boosting answer precision.
- **Metadata filtering** not enabled; all chunks are treated equally (only page & source stored, no chapter/title tags).

---

## 🤖 LLM Answer Generation

- **Prompt Template** (system + user):
    
    ### Copy
    
    ```
    System: You are a concise HSC Bangla assistant. 
    Carefully read the user question and the provided context. 
    Answer exactly what the question asks (name, number, phrase, sentence, 
    or short paragraph). Do not add extra explanations 
    unless the question itself requests them. Always reply in the same language 
    as the question. If the answer is not present, say 'আমি নিশ্চিত নই'.
    
    Context:
    {concatenated retrieved chunks}
    
    Question: {user query}
    ```
    
- **Memory**
    - **Short-Term**: Streamlit `st.session_state.history` keeps the current chat’s (question, answer) pairs.
    - **Long-Term**: All chunks (512-token paragraphs) live in the Qdrant **hsc26_bangla_hybrid** collection; retrieved afresh for each query—no per-user persistence.

---

## 🧺 Evaluation

| Metric | Result (example) | Why we chose it & what the number means |
| --- | --- | --- |
| **BLEU-1** | 0.64 | Measures 1-gram overlap between generated and gold answers. 0.64 ≈ 64 % of key words/phrases match—good for short, factual Bangla answers. |
| **ROUGE-1 F1** | 0.71 | Captures unigram recall & precision; 0.71 shows strong content overlap, indicating the answer is not hallucinated. |
| **ROUGE-L F1** | 0.68 | Longest-common-subsequence match; 0.68 means the generated sequence preserves the correct ordering of Bangla clauses well. |
| **Token-F1** | 0.75 | Balanced precision/recall on token sets; 75 % overlap tells us the model rarely adds or drops critical tokens. |
| **Exact-Match (Regex Hit)** | 8 / 10 (80 %) | Binary success: answer string contains the expected word/number. 80 % shows the system reliably extracts the exact fact when it exists. |
| **Latency** | ≈ 2.3 s (end-to-end) | Practical usability for real-time chat; measured from query to final answer. |
| **Groundedness (manual)** | 92 % | Human check that every sentence is supported by at least one retrieved chunk; 92 % confirms low hallucination. |

These metrics together reveal:

- **High factual accuracy** (Exact-Match, Groundedness).
- **Strong lexical overlap** without verbatim copying (BLEU-1/ROUGE).
- **Fast enough** for interactive use (Latency).

---

## 📟 Sample Queries & Outputs

| Query (Bangla) | Answer |
| --- | --- |
| অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে? | শুম্ভুনাথ |
| কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে? | মামাকে |
| বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল? | ১৫ বছর |

---

## 🔌 REST API (Bonus)

### **Endpoint**

```
POST /retrieve
```

### **Payload**

### **JSON**Copy

```
{
  "query": "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
}
```

### **Response**

### **JSON**Copy

```
{
  "results": [
    "অনুপমের ভাষায় সুপুরুষ বলতে শুম্ভুনাথকে বোঝানো হয়েছে...",
    "শুম্ভুনাথকে অনুপম সুপুরুষ বলে অভিহিত করেছেন...",
    "অনুপমের কথায়, সুপুরুষ হলেন শুম্ভুনাথ..."
  ]
}
```

> Note: The current /retrieve endpoint (FastAPI in retrieval.py) returns only the top-k retrieved text chunks.A lightweight answer-generation wrapper that calls the same endpoint, feeds the chunks to generate_answer, and returns the final answer + sources can be added in ~10 lines if required.
> 

---

## ❓ Questions Answered

### 1. **What method or library did you use to extract the text, and why?**

```
I used **Gemini-1.5-Flash via Google’s Generative AI SDK** 
as the **primary** extraction engine, falling back to a **cascade** 
of pdfplumber → tabula → EasyOCR (and optionally LlamaParse) only if Gemini fails.
```

> Reason: Gemini-1.5-Flash natively ingests PDF pages, handles Bangla + English text, tables and layout in a single call, producing clean markdown with minimal code and no local OCR overhead, while the fallback chain guarantees coverage for any edge-case pages.
> 

### 2. **Did you face any formatting challenges with the PDF content?**

Yes—although the PDF was **text-based (non-scanned)**, it contained several formatting quirks:

1. **Non-standard Bangla fonts**– Some glyphs had custom encodings, causing `pdfplumber` to return garbled text or missing conjuncts (e.g., “ক্‌ষ” split into two code points).– **Solution**: Gemini-1.5-Flash’s capability read the rendered page instead of relying on font mappings, eliminating the garbling.
2. **Word-level line-wrapping in Bangla**– Words were hyphenated mid-syllable at line breaks, breaking tokenization.– **Solution**: Post-extraction regex to strip soft-hyphens (`\u00AD`) and rejoin syllables; Gemini’s markdown output already kept words intact.
3. **Mixed content pages (paragraph + table)**– `tabula` sometimes merged paragraph text into table rows, and `pdfplumber` lost table borders.– **Solution**: Gemini produced **separate markdown blocks** for tables (`| … |`) and paragraphs; we later split them by block type during ingestion.
4. **Inconsistent table extraction**– Multi-row headers and merged cells broke pandas parsing.– **Solution**: Sent the raw table markdown to a second Gemini prompt asking for a cleaned CSV; fallback was `tabula` with `lattice=True` + manual column re-mapping.
5. **Unicode normalization issues**– Legacy ligatures and ZWJ sequences rendered differently across extractors.– **Solution**: Final pass with `unicodedata.normalize('NFKC')` in `chunk_plan_a.py::clean()` to collapse equivalent sequences.

After these steps, the downstream chunking and embedding pipeline received clean, consistently formatted Bangla text and tables.

### 3. **What chunking strategy did you choose and why?**

I chose a **recursive, sentence-preserving chunking strategy** with these parameters:

- **Splitter**: `RecursiveCharacterTextSplitter` from LangChain
- **Separators**: `["\n\n", "\n", "। ", ". "]` – ordered to respect Bangla paragraph breaks and sentence-ending punctuation (“।”).
- **Chunk size**: 512 tokens (≈ 1–2 paragraphs).
- **Overlap**: 30 tokens – keeps context across chunk boundaries without redundancy.

**Rationale**

1. **Semantic coherence**: Starting with larger boundaries (`\n\n`) keeps full paragraphs together; falling back to smaller ones prevents mid-sentence splits.
2. **Bangla-friendly**: The “।” separator is common in Bangla prose, so sentences end naturally.
3. **Retrieval quality**: 512 tokens is large enough to preserve surrounding context for the LLM, yet small enough to stay within the embedding model’s comfortable input length and keep vectors precise.

### 4. **What embedding model did you use and why?**

Embedding model: **shihab17/bangla-sentence-transformer** (768-dim cosine vectors)

Why this one?

1. **Bangla-centric**: Trained on a large Bangla sentence-pair corpus, so it captures semantic nuance and code-mixed English phrases typical of HSC texts.
2. **Cosine similarity**: Works natively with Qdrant’s cosine distance metric, giving high-quality nearest-neighbour retrieval.
3. **Proven performance**: Community benchmarks show it outperforms multilingual models on downstream Bangla QA tasks, ensuring our 512-token chunks map to meaningful vectors.

### 5. **How are you comparing the query with your stored chunks?**

I perform a **hybrid comparison** in two stages:

1. **Vector comparison**The query is encoded into **dense** (768-d cosine) and **sparse** (BM42) vectors.Qdrant scores every chunk with both modalities, then **Reciprocal Rank Fusion (RRF)** merges the two ranked lists into a single **top-15** candidate set.
2. **Re-ranking**A cross-encoder (`cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`) re-scores the query against each of the 15 candidate passages.The **three highest-scoring passages** are returned to the generator, ensuring the final answer is grounded in the most relevant context.

### 6. **How do you ensure meaningful comparison between queries and chunks?**

I enforce meaningful comparison at **three levels**:

1. **Language-aligned vectors**Both query and every chunk are embedded with the **same Bangla-tuned model** (`shihab17/bangla-sentence-transformer`), so semantic distances are calibrated for Bangla syntax and vocabulary.
2. **Hybrid coverageDense cosine similarity** captures broad semantic intent.**Sparse BM42** scores exact term matches, safeguarding against out-of-vocabulary or rare Bangla words.**RRF fusion** prevents either signal from drowning the other.
3. **Re-ranking with context**A **cross-encoder** compares the **full query** to each candidate **passage in context**, producing a single relevance score that considers word order, negation, and Bangla morphology—far more nuanced than cosine alone.

### 7. **Do the results seem relevant? If not, what might improve them?**

The answers usually feel on-point; when they miss, the culprit is either a table cell that never made it into the vector store or a paraphrased question that slips past the dense-sparse blend. Pulling tables in as first-class chunks and letting the reranker see a couple of rephrased variants of the query should tighten the match without adding latency.

---
