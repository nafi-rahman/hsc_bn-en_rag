#src/data_processing/gemini_raw_extract.py  
"""
Hybrid PDF → JSONL extractor
- Gemini 1.5 Flash as the preferred engine
- open-once PDF reuse
- memoised OCR model
- thread-safe tqdm
- engine tag for debugging
"""
import os
import json
import tempfile
import base64
import google.generativeai as genai
import fitz
import pdfplumber
import tabula
import easyocr
from tqdm import tqdm

# -------------------- config --------------------
GEMINI_KEY = os.getenv("GEMINI_API_KEY")           # <— set this
LLAMA_KEY  = os.getenv("LLAMA_CLOUD_API_KEY")

if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    GEMINI_MODEL = genai.GenerativeModel("gemini-1.5-flash")

# -------------------- singletons --------------------
_OCR_READER = None

def get_ocr_reader():
    global _OCR_READER
    if _OCR_READER is None:
        _OCR_READER = easyocr.Reader(["bn", "en"], gpu=False)
    return _OCR_READER

# -------------------- helpers --------------------
def is_text_heavy(page: fitz.Page, threshold: float = 0.05) -> bool:
    text_area = sum(blk["bbox"][2] * blk["bbox"][3]
                    for blk in page.get_text("dict")["blocks"])
    total_area = page.rect.width * page.rect.height
    return (text_area / total_area) > threshold

def ocr_easyocr(img) -> str:
    reader = get_ocr_reader()
    return "\n".join(reader.readtext(img, detail=0))

def llama_parse_page(pdf_path: str, page_idx: int) -> list[str]:
    """Original LlamaParse single-page helper kept for fallback."""
    if not LLAMA_KEY:
        raise RuntimeError("Llama key missing")
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        doc = fitz.open(pdf_path)
        single = fitz.open()
        single.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
        single.save(tmp.name)
        from llama_parse import LlamaParse
        docs = LlamaParse(result_type="markdown").load_data(tmp.name)
        return [d.text for d in docs]

# -------------------- Gemini wrapper --------------------
def gemini_extract_page(pdf_path: str, page_idx: int) -> list[str]:
    """
    Send one PDF page to Gemini 1.5 Flash and ask for plain text/markdown.
    Returns list of text chunks (usually 1 item).
    """
    with tempfile.NamedTemporaryFile(suffix=".pdf") as tmp:
        doc = fitz.open(pdf_path)
        single = fitz.open()
        single.insert_pdf(doc, from_page=page_idx, to_page=page_idx)
        single.save(tmp.name)

        with open(tmp.name, "rb") as f:
            pdf_bytes = f.read()

    prompt = (
        "Extract all text and tables from this PDF page as clean markdown. "
        "Preserve reading order and table structure. Return only markdown."
    )
    response = GEMINI_MODEL.generate_content(
        [
            prompt,
            {"mime_type": "application/pdf", "data": base64.b64encode(pdf_bytes).decode()}
        ]
    )
    # Gemini may return empty candidates on failure
    if not response.candidates:
        raise RuntimeError("Gemini gave empty response")

    text = response.text or ""
    return [text] if text.strip() else []

# -------------------- page-level extraction --------------------
def extract_page(
    pdf_path: str,
    page_idx: int,
    plum_doc,  # pdfplumber doc (already open)
    fitz_doc,  # fitz doc (already open)
) -> list[dict]:
    """Return list of JSONL-ready dicts for one page."""
    page = fitz_doc[page_idx]

    # 0️⃣  FIRST try Gemini if key is present
    if GEMINI_KEY:
        try:
            gem_chunks = gemini_extract_page(pdf_path, page_idx)
            return [
                {
                    "text": txt,
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "page_number": page_idx + 1,
                        "type": "narrativetext",
                        "engine": "gemini-1.5-flash",
                    },
                }
                for txt in gem_chunks
            ]
        except Exception:
            pass  # fall through to legacy chain

    # 1️⃣  Legacy: text-heavy native -> LlamaParse -> EasyOCR
    if is_text_heavy(page):
        try:
            plumb_page = plum_doc.pages[page_idx]
            text = plumb_page.extract_text() or ""
            tables = tabula.read_pdf(
                pdf_path,
                pages=page_idx + 1,
                multiple_tables=True,
                pandas_options={"header": 0},
                silent=True,
            )
            chunks = []
            if text.strip():
                chunks.append(
                    {
                        "text": text.strip(),
                        "metadata": {
                            "source": os.path.basename(pdf_path),
                            "page_number": page_idx + 1,
                            "type": "narrativetext",
                            "engine": "pdfplumber",
                        },
                    }
                )
            for df in tables:
                tbl_csv = df.to_csv(index=False)
                tbl_html = df.to_html(index=False)
                chunks.append(
                    {
                        "text": tbl_csv,
                        "html": tbl_html,
                        "metadata": {
                            "source": os.path.basename(pdf_path),
                            "page_number": page_idx + 1,
                            "type": "table",
                            "engine": "pdfplumber",
                        },
                    }
                )
            return chunks
        except Exception:
            pass

    if LLAMA_KEY and is_text_heavy(page):
        try:
            llama_chunks = llama_parse_page(pdf_path, page_idx)
            return [
                {
                    "text": txt,
                    "metadata": {
                        "source": os.path.basename(pdf_path),
                        "page_number": page_idx + 1,
                        "type": "narrativetext",
                        "engine": "llama-parse",
                    },
                }
                for txt in llama_chunks
            ]
        except Exception:
            pass

    # final fallback: local OCR
    img = np.frombuffer(
        page.get_pixmap(matrix=fitz.Matrix(2, 2)).samples,
        dtype=np.uint8,
    ).reshape(int(page.rect.height * 2), int(page.rect.width * 2), 3)
    ocr_text = ocr_easyocr(img)
    return [
        {
            "text": ocr_text,
            "metadata": {
                "source": os.path.basename(pdf_path),
                "page_number": page_idx + 1,
                "type": "narrativetext",
                "engine": "easyocr",
            },
        }
    ]

# -------------------- driver --------------------
def hybrid_extract(pdf_path: str, out_jsonl: str):
    os.makedirs(os.path.dirname(out_jsonl), exist_ok=True)

    with fitz.open(pdf_path) as fitz_doc, pdfplumber.open(pdf_path) as plum_doc:
        all_chunks = []
        for idx in tqdm(
            range(len(fitz_doc)),
            desc="Processing pages",
            position=0,
            leave=True,
        ):
            all_chunks.extend(
                extract_page(pdf_path, idx, plum_doc=plum_doc, fitz_doc=fitz_doc)
            )

    with open(out_jsonl, "w", encoding="utf-8") as f:
        for c in all_chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")

    print(f"Hybrid extraction complete → {out_jsonl} ({len(all_chunks)} elements)")

# -------------------- CLI entry --------------------
if __name__ == "__main__":
    hybrid_extract("data/00_raw_pdf/raw.pdf", "data/processed/hybrid_output.jsonl")