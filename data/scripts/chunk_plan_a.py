#!/usr/bin/env python3
"""
chunk_plan_a_v2.py
Reads  : hybrid_output.jsonl
Writes : chunks_qdrant.jsonl  (pruned metadata)
Plan-A : 512-token paragraph chunks, 30-token overlap
"""
import json, re, unicodedata, os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from tqdm import tqdm
import pathlib

# ---------- CONFIG ----------
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
RAW_FILE = ROOT / "data" / "processed" / "hybrid_output.jsonl"
OUT_FILE = ROOT / "data" / "processed" / "chunks_a_output.jsonl"
CHUNK_SIZE = 512
OVERLAP    = 30
# ----------------------------

def clean(text: str) -> str:
    """Strip markdown fences, page headers, MCQ tables, extra newlines."""
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"```markdown|```", "", text)
    text = re.sub(r"^\|.*\|.*\|$", "", text, flags=re.MULTILINE)  # drop MCQ tables
    text = re.sub(r"^10 MINUTE SCHOOL.*\n", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()

# Token counter via tiktoken (good proxy)
try:
    import tiktoken
    enc = tiktoken.get_encoding("cl100k_base")
    def tk_len(text: str) -> int:
        return len(enc.encode(text))
except ImportError:
    # Fallback: character count
    tk_len = len

splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", "। ", ". "],
    chunk_size=CHUNK_SIZE,
    chunk_overlap=OVERLAP,
    length_function=tk_len,
)

def main():
    os.makedirs(os.path.dirname(OUT_FILE), exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as fout:
        with open(RAW_FILE, encoding="utf-8") as fin:
            for idx_line, line in enumerate(tqdm(fin, desc="Chunking")):
                if not line.strip():
                    continue
                raw = json.loads(line)
                text = clean(raw["text"])
                meta = raw.get("metadata", {})
                # Pruned metadata
                pruned_meta = {
                    "page": meta.get("page_number"),
                    "source": meta.get("source", "raw.pdf")
                }
                for idx_chunk, ch in enumerate(splitter.split_text(text)):
                    payload = {
                        "id": f"{idx_line}_{idx_chunk}",
                        "text": ch,
                        "metadata": pruned_meta
                    }
                    fout.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"✅ Clean chunks saved → {OUT_FILE}")

if __name__ == "__main__":
    main()