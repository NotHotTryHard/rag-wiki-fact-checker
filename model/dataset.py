import os
import numpy as np
from datasets import load_from_disk, concatenate_datasets
from sentence_transformers import SentenceTransformer
from torch.utils.data import Dataset
from transformers import AutoTokenizer
import sqlite3

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "rag"))
from rag_config import DATA_DIR, DB_PATH, EMBEDDER, EMBEDDER_TAG, QUERY_PROMPT, ENCODE_BATCH

FEVER_DIR = os.path.join(DATA_DIR, "datasets", "fever")
CACHE_DIR = os.path.join(DATA_DIR, "fever_embeddings", EMBEDDER_TAG)


def fever_embeddings(split="train", device="cpu", batch_size=128):
    fever_ds = load_from_disk(FEVER_DIR)

    if split == 'train':
        fever = concatenate_datasets([fever_ds['train'], fever_ds['validation']])
    elif split == 'test':
        fever = fever_ds['test']

    claims = [claim for claim in fever]

    cache_path = os.path.join(CACHE_DIR, f"{split}.npz")
    if os.path.exists(cache_path):
        data = np.load(cache_path)
        return claims, data["embeddings"]

    model = SentenceTransformer(
        EMBEDDER, device=device, model_kwargs={"dtype": "auto"}
    )
    texts = [claim["claim"] for claim in claims]
    embeddings = model.encode(
        texts,
        prompt_name=QUERY_PROMPT,
        normalize_embeddings=True,
        batch_size=batch_size,
        show_progress_bar=True,
    ).astype(np.float32)

    os.makedirs(CACHE_DIR, exist_ok=True)
    np.savez(cache_path, embeddings=embeddings)
    return claims, embeddings


LABEL2ID = {"SUPPORTS": 0, "REFUTES": 1, "NOT ENOUGH INFO": 2}


class ClaimDataset(Dataset):
    def __init__(self, split="train", tokenizer_name="bert-base-uncased",
                 search_index=None, top_k=5, max_length=512, device="cpu"):
        self.top_k = top_k
        self.max_length = max_length
        self.claims, self.embeddings = fever_embeddings(split, device, ENCODE_BATCH)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.search_index = search_index
        self._conn = None

    @property
    def conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(DB_PATH)
        return self._conn

    def __len__(self):
        return len(self.claims)

    def __getitem__(self, idx):
        claim = self.claims[idx]
        query_emb = self.embeddings[idx].reshape(1, -1)
        _, ids_found = self.search_index.search(query_emb, self.top_k)
        chunk_ids = [i for i in ids_found[0].tolist() if i >= 0]

        evidence_texts = []
        if chunk_ids:
            ph = ",".join("?" * len(chunk_ids))
            rows = self.conn.execute(
                f"SELECT id, chunk FROM chunks WHERE id IN ({ph})", chunk_ids
            ).fetchall()
            id_to_text = {r[0]: r[1] for r in rows}
            evidence_texts = [id_to_text.get(i, "") for i in chunk_ids]

        while len(evidence_texts) < self.top_k:
            evidence_texts.append("")

        pairs = self.tokenizer(
            [claim["claim"]] * self.top_k,
            evidence_texts,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        ) # may be i will add distsm but not for now 

        label = LABEL2ID[claim["label"]]
        return pairs["input_ids"], pairs["attention_mask"], label
