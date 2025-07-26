#!/usr/bin/env python3
"""
Bangla hybrid (dense + sparse) → optional MMR → Bangla-friendly re-rank
"""

from typing import List, Dict
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain.schema import Document
from langchain_qdrant.qdrant import QdrantVectorStore, RetrievalMode
from langchain_qdrant.fastembed_sparse import FastEmbedSparse
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    NamedVector,
    NamedSparseVector,
    SparseVector as QdrantSparseVector,
)

# ---------------- CONFIG FLAGS ----------------
QDRANT_URL   = "http://localhost:6333"
COLLECTION   = "hsc26_bangla_hybrid"
DENSE_MODEL  = "shihab17/bangla-sentence-transformer"
SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
RERANK_MODEL = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

FETCH_K      = 30   # candidates per modality
K_AFTER_RRF  = 15   # after fusion
K_FINAL      = 3    # final answer
USE_MMR      = False # toggle MMR
LAMBDAMULT   = 0.4  # only if USE_MMR
# ----------------------------------------------

# 1. embedders & reranker
dense_emb  = HuggingFaceEmbeddings(model_name=DENSE_MODEL)
sparse_emb = FastEmbedSparse(model_name=SPARSE_MODEL)
reranker   = CrossEncoder(RERANK_MODEL)

# 2. low-level client
client = QdrantClient(url=QDRANT_URL)


# --------------------------------------------------
# 3. Hybrid RRF (Reciprocal Rank Fusion)
# --------------------------------------------------
def hybrid_rrf(query: str, top_k: int) -> List[Document]:
    """Return fused Document list, never empty."""
    dense_vec  = dense_emb.embed_query(query)
    sparse_vec = sparse_emb.embed_query(query)
    sparse_dict = sparse_vec.model_dump()

    # dense
    dense_hits = client.search(
        collection_name=COLLECTION,
        query_vector=NamedVector(name="dense", vector=dense_vec),
        limit=FETCH_K,
        with_payload=True,
    )

    # sparse
    sparse_hits = client.search(
        collection_name=COLLECTION,
        query_vector=NamedSparseVector(
            name="bm42",
            vector=QdrantSparseVector(**sparse_dict)
        ),
        limit=FETCH_K,
        with_payload=True,
    )

    # fusion
    rank_map: Dict[str, float] = {}
    for rank, hit in enumerate(dense_hits, 1):
        rank_map[hit.id] = rank_map.get(hit.id, 0) + 1 / (60 + rank)
    for rank, hit in enumerate(sparse_hits, 1):
        rank_map[hit.id] = rank_map.get(hit.id, 0) + 1 / (60 + rank)

    docs = []
    for _id, _ in sorted(rank_map.items(), key=lambda x: x[1], reverse=True)[:top_k]:
        for hit in dense_hits + sparse_hits:
            if hit.id == _id:
                text = hit.payload.get("text", "")
                if text.strip():
                    docs.append(Document(page_content=text, metadata=hit.payload.get("metadata", {})))
                break
    return docs


# --------------------------------------------------
# 4. Retrieval pipeline
# --------------------------------------------------
def retrieve(query: str) -> List[str]:
    docs = hybrid_rrf(query, K_AFTER_RRF)
    if not docs:
        print("⚠️  No documents returned from Qdrant.")
        return []

    # optional MMR
    if USE_MMR:
        store = QdrantVectorStore.from_existing_collection(
            collection_name=COLLECTION,
            embedding=dense_emb,
            sparse_embedding=sparse_emb,
            retrieval_mode=RetrievalMode.HYBRID,
            vector_name="dense",
            sparse_vector_name="bm42",
            url=QDRANT_URL,
        )
        docs = store.max_marginal_relevance_search(
            query=query,
            k=K_FINAL,
            fetch_k=len(docs),
            lambda_mult=LAMBDAMULT,
        )

    # rerank
    pairs = [(query, d.page_content) for d in docs]
    scores = reranker.predict(pairs)
    ranked = [txt for _, txt in sorted(zip(scores, [d.page_content for d in docs]), reverse=True)]
    return ranked[:K_FINAL]


# --------------------------------------------------
# 5. Quick test
# --------------------------------------------------
if __name__ == "__main__":
    q = "অনুপমের ভাষায় সুপুরুষ কাকে বলা হয়েছে?"
    results = retrieve(q)
    if not results:
        print("❌ No answer found.")
    else:
        for idx, txt in enumerate(results, 1):
            print(f"{idx}. {txt}")


# ---------- NEW: FastAPI adapter ----------
from fastapi import FastAPI
from pydantic import BaseModel

class QueryIn(BaseModel):
    query: str

class QueryOut(BaseModel):
    results: List[str]

app = FastAPI()

@app.post("/retrieve", response_model=QueryOut)
def retrieve_endpoint(payload: QueryIn):
    results = retrieve(payload.query)   # existing retrieve() from test.py
    return QueryOut(results=results)