"""Benchmark Recall@K on FEVER dev for FAISS indices.

Usage:
    python rag/benchmark.py --gpu 0
"""

import os
import sqlite3
import numpy as np
import faiss
import torch
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import (
    DATA_DIR, DB_PATH, EMBEDDER, QUERY_PROMPT, INDICES_DIR,
    PQ_VARIANTS, NPROBE, faiss_path, parse_gpu_args,
)

FEVER_DIR = os.path.join(DATA_DIR, "datasets", "fever")

args, device = parse_gpu_args()
print(f"Device: {device}")

model = SentenceTransformer(
    EMBEDDER, device=device, model_kwargs={"torch_dtype": torch.float16}
)
dim = model.get_sentence_embedding_dimension()

# load indices
loaded = {}
for name in PQ_VARIANTS:
    p = faiss_path(name)
    if not os.path.exists(p):
        continue
    idx = faiss.read_index(p)
    faiss.extract_index_ivf(idx).nprobe = NPROBE
    loaded[name] = idx
    print(f"Loaded {name}: {idx.ntotal:,} vectors")

if not loaded:
    print(f"No indices found in {INDICES_DIR}")
    exit(1)

# load fever dataset
fever_ds = load_from_disk(FEVER_DIR)
fever = fever_ds["validation"] if isinstance(fever_ds, dict) else fever_ds
claims = [
    ex for ex in fever
    if ex["label"] != "NOT ENOUGH INFO" and ex.get("evidence_wiki_url")
]
print(f"FEVER claims with evidence: {len(claims)}")

# ── Evaluate ───────────────────────────────────────────────────
conn = sqlite3.connect(DB_PATH)
TOP_K = 5
results = {name: {1: 0, 3: 0, 5: 0} for name in loaded}
total_eval = 0

for ex in tqdm(claims, desc="Evaluating"):
    gold_title = ex["evidence_wiki_url"].replace("_", " ")

    q_vec = model.encode(
        [ex["claim"]], prompt_name=QUERY_PROMPT, normalize_embeddings=True,
    ).astype(np.float32)

    for name, idx in loaded.items():
        _, ids_found = idx.search(q_vec, TOP_K)
        retrieved_ids = [i for i in ids_found[0].tolist() if i >= 0]
        if not retrieved_ids:
            continue

        placeholders = ",".join("?" * len(retrieved_ids))
        rows = conn.execute(
            f"SELECT id, title FROM chunks WHERE id IN ({placeholders})",
            retrieved_ids,
        ).fetchall()
        id_to_title = {r[0]: r[1] for r in rows}
        ranked_titles = [id_to_title.get(i, "") for i in retrieved_ids]

        for k in results[name]:
            if gold_title in ranked_titles[:k]:
                results[name][k] += 1

    total_eval += 1

conn.close()

print()
print("=" * 60)
print(f"RETRIEVAL BENCHMARK — {EMBEDDER}")
print("=" * 60)
print(f"  dim={dim}  nprobe={NPROBE}  claims={total_eval}")
print()

for name, idx in loaded.items():
    size_gb = os.path.getsize(faiss_path(name)) / 1e9
    print(f"  --- {name} ({PQ_VARIANTS[name]}) ---")
    print(f"    Vectors: {idx.ntotal:,}   Size: {size_gb:.2f} GB")
    for k, h in results[name].items():
        pct = h / total_eval * 100 if total_eval else 0
        print(f"    Recall@{k}: {pct:.1f}%  ({h}/{total_eval})")
    print()

print("=" * 60)
