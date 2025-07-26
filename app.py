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
# 3.  Evaluate Tab  (light-weight & crash-safe)
# --------------------------------------------------
with eval_tab:
    st.header("🧪 Evaluation Panel")

    import re, time, traceback
    from typing import List, Tuple

    # lightweight metrics
    import nltk
    from nltk.translate.bleu_score import sentence_bleu
    from rouge_score import rouge_scorer

    # one-time NLTK download (quiet)
    try:
        nltk.data.find("tokenizers/punkt")
    except LookupError:
        nltk.download("punkt", quiet=True)

    # small, fast ROUGE scorer
    rouge = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)

    TESTS: List[Tuple[str, str]] = [
        ("অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?", "শুম্ভুনাথ"),
        ("কাকে অনুপমের ভাগ্য দেবতা বলে উল্লেখ করা হয়েছে?", "মামাকে"),
        ("বিয়ের সময় কল্যাণীর প্রকৃত বয়স কত ছিল?", "১৫ বছর"),
    ]

    def tokenize(text: str) -> List[str]:
        return nltk.word_tokenize(text.lower())

    def lex_metrics(expected: str, generated: str) -> dict:
        gold_tok = tokenize(expected)
        pred_tok = tokenize(generated)

        # BLEU-1
        bleu1 = sentence_bleu([gold_tok], pred_tok, weights=(1, 0, 0, 0))

        # ROUGE
        rouges = rouge.score(expected.lower(), generated.lower())

        # token-level F1
        overlap = set(pred_tok) & set(gold_tok)
        p = len(overlap) / max(len(set(pred_tok)), 1)
        r = len(overlap) / max(len(set(gold_tok)), 1)
        f1 = 2 * p * r / max(p + r, 1e-9)

        return {
            "BLEU-1": round(bleu1, 3),
            "ROUGE-1": round(rouges["rouge1"].fmeasure, 3),
            "ROUGE-L": round(rouges["rougeL"].fmeasure, 3),
            "Token-F1": round(f1, 3),
        }

    if st.button("Run All Tests"):
        try:
            results = []
            for q, expected in TESTS:
                start = time.perf_counter()

                # retrieve
                resp = requests.post(RETRIEVER_URL, json={"query": q}, timeout=30)
                docs = resp.json()["results"]

                # generate
                from rag.generator import generate_answer
                ans = generate_answer(q, [RetrievedChunk(text=t) for t in docs])

                elapsed = (time.perf_counter() - start) * 1000  # ms

                # metrics
                m = lex_metrics(expected, ans.text)
                m.update(
                    {
                        "Question": q,
                        "Expected": expected,
                        "Generated": ans.text,
                        "Latency(ms)": round(elapsed, 1),
                        "Answer Tokens": len(ans.text.split()),
                        "Regex Hit": bool(re.search(re.escape(expected), ans.text, flags=re.I)),
                    }
                )
                results.append(m)

            st.dataframe(results)

            exact = sum(r["Regex Hit"] for r in results)
            st.metric("Exact-Match", f"{exact}/{len(results)} ({exact/len(results):.2%})")
            st.metric("Avg Token-F1", f"{np.mean([r['Token-F1'] for r in results]):.3f}")
            st.metric("Avg BLEU-1", f"{np.mean([r['BLEU-1'] for r in results]):.3f}")
            st.metric("Avg ROUGE-1", f"{np.mean([r['ROUGE-1'] for r in results]):.3f}")

        except Exception as e:
            st.error("Evaluation failed – continuing safely.")
            with st.expander("Show technical details"):
                st.code(traceback.format_exc())