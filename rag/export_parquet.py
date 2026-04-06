"""Export chunk+embedding pairs as Parquet shards directly to HuggingFace.

Usage:
    python rag/export_parquet.py --embedder harrier-oss-v1-270m --repo username/wiki-en-harrier-270m
    python rag/export_parquet.py --embedder harrier-oss-v1-0.6b --repo username/wiki-en-harrier-0.6b
"""

import os
import sys
import io
import glob
import argparse
import sqlite3
import pyarrow as pa
import pyarrow.parquet as pq
from safetensors.numpy import load_file
from huggingface_hub import HfApi

sys.path.insert(0, os.path.dirname(__file__))
from rag_config import DATA_DIR, DB_PATH

parser = argparse.ArgumentParser()
parser.add_argument("--embedder", required=True, help="Embedder tag, e.g. harrier-oss-v1-270m")
parser.add_argument("--repo", required=True, help="HF repo id, e.g. username/wiki-en-harrier-270m")
args = parser.parse_args()

emb_dir = os.path.join(DATA_DIR, "embeddings", args.embedder)
shard_files = sorted(glob.glob(os.path.join(emb_dir, "shard_*.safetensors")))
if not shard_files:
    print(f"No shards in {emb_dir}")
    exit(1)

api = HfApi()
api.create_repo(args.repo, repo_type="dataset", exist_ok=True)

n_shards = len(shard_files)
conn = sqlite3.connect(DB_PATH)

for si, sf in enumerate(shard_files):
    hf_path = f"data/train-{si:05d}-of-{n_shards:05d}.parquet"
    print(f"[{si+1}/{n_shards}] {sf} -> {hf_path}")

    data = load_file(sf)
    ids = data["ids"]
    vecs = data["vecs"]
    id_list = ids.tolist()
    n = len(id_list)

    BATCH = 10_000
    id_to_row = {}
    for b in range(0, n, BATCH):
        batch = id_list[b:b + BATCH]
        placeholders = ",".join("?" * len(batch))
        rows = conn.execute(
            f"SELECT id, title, chunk FROM chunks WHERE id IN ({placeholders})",
            batch,
        ).fetchall()
        id_to_row.update({r[0]: (r[1], r[2]) for r in rows})

    titles = []
    chunks = []
    for i in id_list:
        t, c = id_to_row.get(i, ("", ""))
        titles.append(t)
        chunks.append(c)

    table = pa.table({
        "chunk_id": pa.array(id_list, type=pa.int64()),
        "article_title": pa.array(titles, type=pa.string()),
        "text": pa.array(chunks, type=pa.string()),
        "embedding": pa.FixedSizeListArray.from_arrays(
            pa.array(vecs.ravel(), type=pa.float32()),
            list_size=vecs.shape[1],
        ),
    })

    buf = io.BytesIO()
    pq.write_table(table, buf, compression="zstd")
    buf.seek(0)

    api.upload_file(
        path_or_fileobj=buf,
        path_in_repo=hf_path,
        repo_id=args.repo,
        repo_type="dataset",
    )
    size_gb = buf.getbuffer().nbytes / 1e9
    print(f"  Uploaded {n:,} rows, {size_gb:.2f} GB")
    del data, ids, vecs, table, buf

conn.close()
print(f"Done. https://huggingface.co/datasets/{args.repo}")
