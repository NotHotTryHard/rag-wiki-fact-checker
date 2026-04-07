import streamlit as st
import os
import sys
import numpy as np
import faiss
import sqlite3
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), "rag"))
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

from rag_config import DB_PATH, EMBEDDER, QUERY_PROMPT, NPROBE, faiss_path
from model_config import MODEL_NAME, PQ_NAME, MAX_LENGTH, NUM_LABELS, TOP_K
from model import TruthfulnessSayer
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download                                                                                                                            
                                                                                                                                                                         
REPO = "NotHotTryHard/fact-checker-data"

@st.cache_resource
def load_all():
    weights = hf_hub_download(REPO, "best.pt", repo_type="dataset")
    index_path = hf_hub_download(REPO, "PQ256.faiss", repo_type="dataset")
    db_path = hf_hub_download(REPO, "wiki_en.db", repo_type="dataset")

    embedder = SentenceTransformer(EMBEDDER)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sayer = TruthfulnessSayer(MODEL_NAME, NUM_LABELS, weights)
    index = faiss.read_index(index_path)
    faiss.extract_index_ivf(index).nprobe = NPROBE
    conn = sqlite3.connect(db_path, check_same_thread=False)
    return embedder, tokenizer, sayer, index, conn


embedder, tokenizer, sayer, index, conn = load_all()

st.title("Fact Checker")
st.write("Enter a claim to check its truthfulness against Wikipedia RAG-retrieved evidence.")

claim = st.text_input("Claim")

if st.button("Check") and claim.strip():
    with st.spinner("Searching..."):
        query_emb = embedder.encode(
            [claim], prompt_name=QUERY_PROMPT, normalize_embeddings=True
        ).astype(np.float32)

        _, ids_found = index.search(query_emb, TOP_K)
        chunk_ids = [i for i in ids_found[0].tolist() if i >= 0]

        evidence_texts = []
        if chunk_ids:
            ph = ",".join("?" * len(chunk_ids))
            rows = conn.execute(
                f"SELECT id, chunk FROM chunks WHERE id IN ({ph})", chunk_ids
            ).fetchall()
            id_to_text = {r[0]: r[1] for r in rows}
            evidence_texts = [id_to_text.get(i, "") for i in chunk_ids]
        while len(evidence_texts) < TOP_K:
            evidence_texts.append("")

        pairs = tokenizer(
            [claim] * TOP_K, evidence_texts,
            truncation=True, max_length=MAX_LENGTH, padding="longest", return_tensors="pt",
        )
        input_ids = pairs["input_ids"].unsqueeze(0)
        attention_mask = pairs["attention_mask"].unsqueeze(0)

        score = sayer(input_ids, attention_mask).item()

    st.metric("Truthfulness", f"{score:.2%}")
    st.progress(score)

    st.subheader("Evidence")
    for i, ev in enumerate(evidence_texts):
        if ev:
            with st.expander(f"Evidence {i+1}: {ev[:100]}..."):
                st.write(ev)
