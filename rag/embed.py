"""Embed chunks in parallel by shards. Each GPU processes its own shard.

Usage:
    python rag/embed.py --gpu 4 --shard 1/4
    python rag/embed.py --gpu 5 --shard 2/4
    python rag/embed.py --gpu 6 --shard 3/4
    python rag/embed.py --gpu 7 --shard 4/4
"""

import os
import sqlite3
import numpy as np
import torch
from tqdm import tqdm
from safetensors.numpy import save_file
from sentence_transformers import SentenceTransformer
from config import DB_PATH, EMBEDDER, EMBEDDINGS_DIR, ENCODE_BATCH, MAX_TOKENS_PER_BATCH,parse_gpu_args

args, device = parse_gpu_args(extra_args=[
    (["--shard"], {"type": str, "required": True, "help": "Shard K/N, e.g. 1/4"}),
])

k, n = map(int, args.shard.split("/"))
k -= 1  # 1-based input → 0-based for modulo
print(f"Device: {device} | Shard {k+1}/{n}")

os.makedirs(EMBEDDINGS_DIR, exist_ok=True)

shard_path = os.path.join(EMBEDDINGS_DIR, f"shard_{k}.safetensors")

if os.path.exists(shard_path):
    print(f"Shard {k} already exists, skipping.")
    exit(0)

model = SentenceTransformer(
    EMBEDDER, device=device, model_kwargs={"torch_dtype": torch.float16}
)
dim = model.get_sentence_embedding_dimension()
print(f"Embedder dim: {dim}")

conn = sqlite3.connect(DB_PATH)
rows = conn.execute(
    "SELECT id, chunk FROM chunks WHERE id % ? = ? ORDER BY id", (n, k)
).fetchall()
conn.close()

total = len(rows)
print(f"Shard {k+1}: {total:,} chunks")

rows.sort(key=lambda r: len(r[1]))

all_ids = np.array([r[0] for r in rows], dtype=np.int64)
all_vecs = np.empty((total, dim), dtype=np.float32)

pbar = tqdm(total=total, desc=f"Shard {k+1} embed")
start = 0
while start < total:
    # приближение: берём самый длинный потенциальный конец батча
    max_len = max(1, len(rows[min(start + MAX_TOKENS_PER_BATCH, total) - 1][1]))
    batch_size = max(1, MAX_TOKENS_PER_BATCH * 4 // max_len)  # 4 символа ≈ 1 токен
    end = min(start + batch_size, total)

    texts = [rows[i][1] for i in range(start, end)]
    vecs = model.encode(
        texts, batch_size=len(texts), normalize_embeddings=True,
    ).astype(np.float32)

    all_vecs[start:end] = vecs
    pbar.update(end - start)
    start = end

save_file({"vecs": all_vecs, "ids": all_ids}, shard_path)
print(f"Saved {shard_path} ({os.path.getsize(shard_path) / 1e9:.1f} GB)")
print("Done.")
