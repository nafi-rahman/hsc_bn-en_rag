# app.py
import os
import json
import requests
import streamlit as st
from typing import List, Dict
import numpy as np
import torch
# ---------- Pydantic models ----------
from pydantic import BaseModel
class RetrievedChunk(BaseModel):
    text: str
    metadata: Dict = {}
class Answer(BaseModel):
    text: str
    sources: List[Dict]

# ---------- Globals ----------
RETRIEVER_URL = "http://localhost:8000/retrieve"

# --------------------------------------------------
# 0.  Ask for API key once per session
# --------------------------------------------------
if "gemini_key" not in st.session_state:
    st.session_state.gemini_key = None
if st.session_state.gemini_key is None:
    key = st.text_input("Enter your Gemini API key:", type="password")
    if st.button("Save key"):
        os.environ["GEMINI_API_KEY"] = key
        st.session_state.gemini_key = key
        st.rerun()
    st.stop()

# --------------------------------------------------
# 1.  Tab layout
# --------------------------------------------------
chat_tab, eval_tab = st.tabs(["💬 Chat", "🧪 Evaluate"])

# --------------------------------------------------
# 2.  Chat Tab
# --------------------------------------------------
with chat_tab:
    st.title("💬 HSC-Bangla RAG Chat")

    # Short-term memory: list of (q, a) pairs
    if "history" not in st.session_state:
        st.session_state.history = []

    for q, a in st.session_state.history:
        st.chat_message("user").write(q)
        st.chat_message("assistant").write(a)

    if prompt := st.chat_input("প্রশ্ন করুন…"):
        st.chat_message("user").write(prompt)

        with st.spinner("অনুসন্ধান ও উত্তর তৈরি হচ্ছে…"):
            # Retrieve
            resp = requests.post(RETRIEVER_URL, json={"query": prompt}, timeout=30)
            if resp.status_code != 200:
                st.error("রিট্রিভাল ব্যর্থ!")
                st.stop()
            docs = resp.json()["results"]

            # Generate
            from rag.generator import generate_answer
            chunks = [RetrievedChunk(text=t) for t in docs]
            answer = generate_answer(prompt, chunks)

        st.chat_message("assistant").write(answer.text)
        st.session_state.history.append((prompt, answer.text))

# --------------------------------------------------
# 3.  Evaluate Tab
# --------------------------------------------------
with eval_tab:
    st.header("🧪 Evaluation Panel")

    import re
    from sentence_transformers import SentenceTransformer
    from sklearn.metrics.pairwise import cosine_similarity

    sim_model = SentenceTransformer(
    "shihab17/bangla-sentence-transformer",
    device="cpu"          # 👈 explicit device
    )

    TESTS = [
        ("অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "শুম্ভুনাথ"),
        ("কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "মামাকে"),
        ("বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "১৫ বছর"),
    ]

    if st.button("Run All Tests"):
        results = []
        for q, expected in TESTS:
            with st.spinner(f"Testing: {q}"):
                # Retrieve
                resp = requests.post(RETRIEVER_URL, json={"query": q}, timeout=30)
                docs = resp.json()["results"]

                # Generate
                from rag.generator import generate_answer
                ans = generate_answer(q, [RetrievedChunk(text=t) for t in docs])

                # 1) Exact via regex
                hit = bool(re.search(re.escape(expected), ans.text, flags=re.I))

                # 2) Groundedness (answer vs each chunk)
                emb_ans = sim_model.encode([ans.text])
                emb_ctx = sim_model.encode(docs)
                groundedness = float(cosine_similarity(emb_ans, emb_ctx).max())

                # 3) Relevance (query vs each chunk)
                emb_qry = sim_model.encode([q])
                relevance = float(cosine_similarity(emb_qry, emb_ctx).mean())

                results.append({
                    "Question": q,
                    "Expected": expected,
                    "Generated": ans.text,
                    "Regex Hit": hit,
                    "Groundedness": round(groundedness, 3),
                    "Relevance": round(relevance, 3),
                })

        st.subheader("📊 Results")
        st.dataframe(results)

        exact = sum(r["Regex Hit"] for r in results)
        st.metric("Regex-Match Accuracy", f"{exact}/{len(results)} ({exact/len(results):.2%})")
        st.metric("Avg Groundedness", f"{np.mean([r['Groundedness'] for r in results]):.3f}")
        st.metric("Avg Relevance", f"{np.mean([r['Relevance'] for r in results]):.3f}")