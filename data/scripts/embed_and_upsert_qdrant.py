#!/usr/bin/env python3
"""
embed_and_upsert_hybrid.py
Dense vectors  → SentenceTransformer  (shihab17/bangla-sentence-transformer)
Sparse vectors → fastembed SparseTextEmbedding  (BM42 – Qdrant/bm42-all-minilm-l6-v2-attentions)
Hybrid upsert  → Qdrant collection "hsc26_bangla_hybrid"
"""
import json, uuid, pathlib
from sentence_transformers import SentenceTransformer
from fastembed import SparseTextEmbedding
from qdrant_client import QdrantClient
from qdrant_client.http.models import (
    Distance,
    VectorParams,
    SparseVectorParams,
    SparseIndexParams,
    PointStruct,
    SparseVector,
)
from tqdm import tqdm

# ---------- CONFIG ----------
ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
CHUNK_FILE   = ROOT / "data" / "processed" / "chunks_a_output.jsonl"
DENSE_MODEL  = "shihab17/bangla-sentence-transformer"
SPARSE_MODEL = "Qdrant/bm42-all-minilm-l6-v2-attentions"
QDRANT_URL   = "http://localhost:6333"
COLLECTION   = "hsc26_bangla_hybrid"
BATCH_SIZE   = 64
# --------------------------------

# 1. Models -------------------------------------------------------------
dense_model  = SentenceTransformer(DENSE_MODEL)
sparse_model = SparseTextEmbedding(model_name=SPARSE_MODEL)

# 2. Qdrant client & collection -----------------------------------------
client = QdrantClient(url=QDRANT_URL)

client.recreate_collection(
    collection_name=COLLECTION,
    vectors_config={
        "dense": VectorParams(size=768, distance=Distance.COSINE)
    },
    sparse_vectors_config={
        "bm42": SparseVectorParams(
            index=SparseIndexParams(on_disk=False)
            # modifier removed – BM42 applies IDF internally
        )
    }
)

# 3. Load chunks --------------------------------------------------------
with open(CHUNK_FILE, encoding="utf-8") as f:
    chunks = [json.loads(line) for line in f]

# 4. Upsert -------------------------------------------------------------
def batched(iterable, n):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == n:
            yield batch
            batch = []
    if batch:
        yield batch

for batch in tqdm(
    batched(chunks, BATCH_SIZE),
    total=(len(chunks) + BATCH_SIZE - 1) // BATCH_SIZE,
    desc="Upserting"
):
    texts = [c["text"] for c in batch]

    # dense vectors
    dense_vecs = dense_model.encode(texts, batch_size=BATCH_SIZE, show_progress_bar=False)

    # sparse vectors
    sparse_vecs = list(sparse_model.embed(texts))

    points = []
    for dense_vec, sparse_vec, chunk in zip(dense_vecs, sparse_vecs, batch):
        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vec.tolist(),
                    "bm42": SparseVector(
                        indices=sparse_vec.indices.tolist(),
                        values=sparse_vec.values.tolist()
                    )
                },
                payload={"text": chunk["text"], **chunk["metadata"]}
            )
        )
    client.upsert(collection_name=COLLECTION, points=points)

print("✅ Hybrid upsert complete")