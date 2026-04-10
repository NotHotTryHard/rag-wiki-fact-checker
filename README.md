---
title: RAG Wiki Fact Checker
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.11.0
app_file: app.py
pinned: false
---
[ОТЧЁТ](report.pdf)

# RAG Wiki Fact Checker

Wikipedia-grounded fact-checking system that retrieves evidence passages and verifies claims with an NLI model.

Given a claim, the pipeline retrieves the top-k most relevant chunks from an English Wikipedia dump via FAISS, then runs a verifier over each `[evidence, claim]` pair. Per-pair logits are **maxpooled** across chunks and the final score is computed as:

```
truthfulness = p(supports) / (p(supports) + p(refutes))
```

## Pipeline

```
claim
  --> encode with harrier-oss-v1-0.6b
  --> FAISS IVF4096+PQ256 search (top-5 chunks)
  --> tokenize 5x [evidence, claim] pairs
  --> verifier model (BERT / DeBERTa)
  --> maxpool logits over 5 pairs
  --> softmax --> truthfulness score
```

## Stack

| Component | Details |
|-----------|---------|
| Embedder | `microsoft/harrier-oss-v1-0.6b` (600M params) |
| Index | FAISS IVF4096 + PQ256, ~6.3 GB, 23.7M vectors |
| Corpus | English Wikipedia, 200-word chunks with 50-word overlap, title prepended |
| Verifier | BERT-base / DeBERTa-v3-base, trained on FEVER |
| DB | SQLite (chunks stored on disk, prefetched with batched reads) |

## Retrieval Benchmark (FEVER, 45k claims)

| Embedder | Index | Size | Recall@1 | Recall@3 | Recall@5 |
|----------|-------|------|----------|----------|----------|
| harrier-270m | PQ160 | 4.00 GB | 32.5% | 44.3% | 47.4% |
| harrier-270m | PQ320 | 7.80 GB | 34.2% | 45.6% | 48.2% |
| harrier-0.6b | PQ128 | 3.25 GB | 32.5% | 44.6% | 47.7% |
| harrier-0.6b | **PQ256** | 6.29 GB | **35.5%** | **47.3%** | **50.0%** |

PQ256 on the 0.6b embedder gives the best recall/size trade-off and is used in the demo.

## Verifier Training

Two base models were trained on FEVER for 5 epochs:

- **BERT-base** (80M) -- val accuracy ~97.9%, batch size 32, ~60k steps
- **DeBERTa-v3-base** (110M) -- val accuracy ~96.6%, batch size 16, ~160k steps

Both results are high due to data leakage in FEVER (models learn claim structure rather than evidence grounding). Augmentation with hard negatives (top-20..25 retrieval chunks as soft negatives) is a planned improvement.

## Running locally

```bash
pip install -r requirements.txt
python app.py
```

The app downloads the FAISS index, SQLite database, and model weights from HuggingFace on first run.

A hosted demo is available at [HuggingFace Spaces](https://huggingface.co/spaces/NotHotTryHard/rag-wiki-fact-checker).

## Project Structure

```
app.py              # Gradio UI + inference pipeline
model/
  model.py          # FactChecker + TruthfulnessSayer (maxpool + scoring)
  model_config.py   # model name, top-k, max length, hyperparams
rag/
  rag_config.py     # embedder name, chunk size, FAISS params
report.pdf          # project report (ru)
```
