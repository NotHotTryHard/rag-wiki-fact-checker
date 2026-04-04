"""Train FAISS index and build it from pre-computed embeddings.

Usage:
    python rag/quantize.py --pq PQ160
"""

import os
import glob
import numpy as np
import faiss
from tqdm import tqdm
from safetensors.numpy import load_file
from config import (
    EMBEDDINGS_DIR, INDICES_DIR, TRAIN_SIZE,
    PQ_VARIANTS, faiss_path, checkpoint_path,
)
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--pq", choices=list(PQ_VARIANTS), required=True)
args = parser.parse_args()

name = args.pq
factory = PQ_VARIANTS[name]
os.makedirs(INDICES_DIR, exist_ok=True)

# loading shards
shard_files = sorted(glob.glob(os.path.join(EMBEDDINGS_DIR, "shard_*.safetensors")))
if not shard_files:
    print(f"No shards found in {EMBEDDINGS_DIR}")
    exit(1)

shards = []
for path in tqdm(shard_files, desc="Loading shards"):
    data = load_file(path)
    shards.append((data["ids"], data["vecs"]))
    tqdm.write(f"  {path}: {len(data['ids']):,} vectors")

total = sum(len(s[0]) for s in shards)
dim = shards[0][1].shape[1]
print(f"Total: {total:,} vectors, dim={dim}")

# train
ckpt = checkpoint_path(name)
if os.path.exists(ckpt):
    print(f"Loading trained index from {ckpt}")
    index = faiss.read_index(ckpt)
else:
    # Sample training vectors from all shards
    per_shard = TRAIN_SIZE // len(shards) + 1
    train_parts = []
    for ids, vecs in shards:
        step = max(1, len(vecs) // per_shard)
        train_parts.append(vecs[::step][:per_shard])
    train_vecs = np.concatenate(train_parts)[:TRAIN_SIZE]
    print(f"Training {name} ({factory}) on {len(train_vecs):,} vectors ...")

    index = faiss.index_factory(dim, factory, faiss.METRIC_INNER_PRODUCT)
    index.train(train_vecs)
    faiss.write_index(index, ckpt)
    print(f"Trained index saved to {ckpt}")
    del train_vecs

# adding vectors to faiss index
pbar = tqdm(total=total, desc="Adding vectors")
for ids, vecs in shards:
    index.add_with_ids(vecs, ids)
    pbar.update(len(ids))
pbar.close()

# saving index
p = faiss_path(name)
faiss.write_index(index, p)
size_gb = os.path.getsize(p) / 1e9
print(f"{name}: {index.ntotal:,} vectors, {size_gb:.2f} GB → {p}")

if os.path.exists(ckpt):
    os.remove(ckpt)

print("Done.")
