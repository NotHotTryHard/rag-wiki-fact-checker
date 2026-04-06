"""Collect retrieval misses from FEVER for analysis.

Stops as soon as N misses are collected (default 100).

Usage:
    python rag/visualise_misses.py --gpu 0 --pq PQ160
    python rag/visualise_misses.py --gpu 0 --pq PQ160 -n 200
"""

import os
import json
import sqlite3
import numpy as np
import faiss
from datasets import load_from_disk
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from rag_config import (
    DATA_DIR, DB_PATH, EMBEDDER, QUERY_PROMPT,
    PQ_VARIANTS, NPROBE, faiss_path, parse_gpu_args,
)

FEVER_DIR = os.path.join(DATA_DIR, "datasets", "fever")

args, device = parse_gpu_args(extra_args=[
    (["--pq"], {"choices": list(PQ_VARIANTS), "required": True}),
    (["-n"], {"type": int, "default": 100, "help": "Number of misses to collect"}),
    (["-o"], {"type": str, "default": None, "help": "Output JSON path"}),
])

name = args.pq
idx = faiss.read_index(faiss_path(name))
faiss.extract_index_ivf(idx).nprobe = NPROBE
print(f"Loaded {name}: {idx.ntotal:,} vectors")

model = SentenceTransformer(EMBEDDER, device=device, model_kwargs={"dtype": "auto"})

fever_ds = load_from_disk(FEVER_DIR)
fever = fever_ds["validation"] if isinstance(fever_ds, dict) else fever_ds
claims = [
    ex for ex in fever
    if ex["label"] != "NOT ENOUGH INFO" and ex.get("evidence_wiki_url")
]
print(f"FEVER claims: {len(claims):,}")

conn = sqlite3.connect(DB_PATH)
TOP_K = 5
misses = []

for ex in tqdm(claims, desc=f"Collecting misses (0/{args.n})"):
    gold_title = ex["evidence_wiki_url"].replace("_", " ")

    q_vec = model.encode(
        [ex["claim"]], prompt_name=QUERY_PROMPT, normalize_embeddings=True,
    ).astype(np.float32)

    _, ids_found = idx.search(q_vec, TOP_K)
    retrieved_ids = [i for i in ids_found[0].tolist() if i >= 0]
    if not retrieved_ids:
        continue

    placeholders = ",".join("?" * len(retrieved_ids))
    rows = conn.execute(
        f"SELECT id, title, chunk FROM chunks WHERE id IN ({placeholders})",
        retrieved_ids,
    ).fetchall()
    id_to_row = {r[0]: {"title": r[1], "chunk": r[2]} for r in rows}
    ranked_titles = [id_to_row.get(i, {}).get("title", "") for i in retrieved_ids]

    if gold_title not in ranked_titles[:TOP_K]:
        misses.append({
            "claim": ex["claim"],
            "label": ex["label"],
            "gold_title": gold_title,
            "retrieved": [
                {"id": i, "title": id_to_row.get(i, {}).get("title", ""),
                 "chunk": id_to_row.get(i, {}).get("chunk", "")[:300]}
                for i in retrieved_ids
            ],
        })
        tqdm.write(f"  miss #{len(misses)}: \"{ex['claim'][:80]}\" — gold: {gold_title}")
        if len(misses) >= args.n:
            break

conn.close()

out_path = args.o or f"misses_{name}.json"
with open(out_path, "w") as f:
    json.dump(misses, f, ensure_ascii=False, indent=2)
print(f"\nSaved {len(misses)} misses to {out_path}")
