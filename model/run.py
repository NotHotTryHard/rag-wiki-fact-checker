"""
Usage:
    python model/run.py --gpu 0 "The Eiffel Tower is located in Berlin"
"""

import os
import sys
import numpy as np
import faiss
import sqlite3

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rag"))
from rag_config import DB_PATH, EMBEDDER, QUERY_PROMPT, NPROBE, faiss_path, parse_gpu_args

from model_config import MODEL_NAME, PQ_NAME, MAX_LENGTH, NUM_LABELS, TOP_K
from model import TruthfulnessSayer

from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer

args, device = parse_gpu_args(extra_args=[
    (["--weights"], {"type": str, "default": None, "help": "Path to model weights"}),
    (["--model"], {"type": str, "default": MODEL_NAME}),
    (["--pq"], {"type": str, "default": PQ_NAME}),
])

weights = args.weights or os.path.join(
    "checkpoints", f"{args.model.rsplit('/', 1)[-1]}__{args.pq}", "best.pt"
)

print("Loading models...")
embedder = SentenceTransformer(EMBEDDER, device=device, model_kwargs={"dtype": "auto"})
tokenizer = AutoTokenizer.from_pretrained(args.model)
model = TruthfulnessSayer(args.model, NUM_LABELS, weights).to(device)

index = faiss.read_index(faiss_path(args.pq))
faiss.extract_index_ivf(index).nprobe = NPROBE
conn = sqlite3.connect(DB_PATH)
print("Ready.\n")

while True:
    try:
        claim = input("Claim: ").strip()
    except (EOFError, KeyboardInterrupt):
        break
    if not claim:
        continue

    query_emb = embedder.encode(
        [claim], prompt_name=QUERY_PROMPT, normalize_embeddings=True
    ).astype(np.float32)

    _, ids_found = index.search(query_emb, TOP_K)
    chunk_ids = [i for i in ids_found[0].tolist() if i >= 0]

    evidence_texts = []
    if chunk_ids:
        ph = ",".join("?" * len(chunk_ids))
        rows = conn.execute(f"SELECT id, chunk FROM chunks WHERE id IN ({ph})", chunk_ids).fetchall()
        id_to_text = {r[0]: r[1] for r in rows}
        evidence_texts = [id_to_text.get(i, "") for i in chunk_ids]
    while len(evidence_texts) < TOP_K:
        evidence_texts.append("")

    pairs = tokenizer(
        [claim] * TOP_K, evidence_texts,
        truncation=True, max_length=MAX_LENGTH, padding="longest", return_tensors="pt",
    )
    input_ids = pairs["input_ids"].unsqueeze(0).to(device)
    attention_mask = pairs["attention_mask"].unsqueeze(0).to(device)

    score = model(input_ids, attention_mask).item()

    print(f"Truthfulness: {score:.4f}")
    for i, ev in enumerate(evidence_texts):
        if ev:
            print(f"  Evidence {i+1}: {ev[:120]}...")
    print()

conn.close()
