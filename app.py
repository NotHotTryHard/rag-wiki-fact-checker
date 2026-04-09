import os
import sys
import sqlite3

import numpy as np
import faiss
import torch
import gradio as gr
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "rag"))
sys.path.append(os.path.join(os.path.dirname(__file__), "model"))

from rag_config import EMBEDDER, QUERY_PROMPT, NPROBE
from model_config import MODEL_NAME, MAX_LENGTH, NUM_LABELS, TOP_K
from model import TruthfulnessSayer
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from huggingface_hub import hf_hub_download


REPO = "NotHotTryHard/fact-checker-data"


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


print("Loading models and index...")
EMBEDDER_MODEL, TOKENIZER, SAYER, INDEX, CONN = load_all()
print("Ready.")


# --------------------------------------------------------------------------
# Core inference
# --------------------------------------------------------------------------
def run_fact_check(claim: str):
    """Returns (score in [0,1], evidence_texts: list[str])."""
    query_emb = EMBEDDER_MODEL.encode(
        [claim],
        prompt_name=QUERY_PROMPT,
        normalize_embeddings=True,
    ).astype(np.float32)

    _, ids_found = INDEX.search(query_emb, TOP_K)
    chunk_ids = [i for i in ids_found[0].tolist() if i >= 0]

    evidence_texts = []
    if chunk_ids:
        ph = ",".join("?" * len(chunk_ids))
        rows = CONN.execute(
            f"SELECT id, chunk FROM chunks WHERE id IN ({ph})", chunk_ids
        ).fetchall()
        id_to_text = {r[0]: r[1] for r in rows}
        evidence_texts = [id_to_text.get(i, "") for i in chunk_ids]

    while len(evidence_texts) < TOP_K:
        evidence_texts.append("")

    pairs = TOKENIZER(
        [claim] * TOP_K,
        evidence_texts,
        truncation=True,
        max_length=MAX_LENGTH,
        padding="longest",
        return_tensors="pt",
    )
    input_ids = pairs["input_ids"].unsqueeze(0)
    attention_mask = pairs["attention_mask"].unsqueeze(0)

    with torch.no_grad():
        score = SAYER(input_ids, attention_mask).item()

    return score, evidence_texts


def score_to_verdict(score: float):
    """Map truthfulness score to (label, color)."""
    if score >= 0.75:
        return "Likely TRUE", "#16a34a"
    if score <= 0.25:
        return "Likely FALSE", "#dc2626"
    return "Uncertain", "#d97706"


READY_HTML = (
    "<div class='verdict-card'>"
    "<div class='verdict-label' style='color:#64748b;'>Ready</div>"
    "<div class='verdict-caption'>Enter a claim and click Check.</div>"
    "</div>"
)


def format_verdict_html(score: float) -> str:
    label, color = score_to_verdict(score)
    pct = f"{score * 100:.1f}%"
    return (
        "<div class='verdict-card'>"
        f"<div class='verdict-label' style='color:{color};'>{label}</div>"
        f"<div class='verdict-score'>{pct}</div>"
        "<div class='verdict-caption'>truthfulness = p(supports) / (p(supports) + p(refutes))</div>"
        "</div>"
    )


def format_evidence_markdown(evidence_texts) -> str:
    blocks = []
    for i, ev in enumerate(evidence_texts, start=1):
        if not ev:
            continue
        blocks.append(f"**Evidence {i}**\n\n{ev}\n\n---")
    return "\n".join(blocks) if blocks else "_No evidence retrieved._"


# --------------------------------------------------------------------------
# Gradio callbacks
# --------------------------------------------------------------------------
def history_to_dataframe(history) -> pd.DataFrame:
    if not history:
        return pd.DataFrame(columns=["#", "Claim", "Verdict", "Score"])
    rows = []
    for i, item in enumerate(reversed(history), start=1):
        claim_preview = item["claim"]
        if len(claim_preview) > 90:
            claim_preview = claim_preview[:87] + "..."
        rows.append(
            {
                "#": i,
                "Claim": claim_preview,
                "Verdict": item["label"],
                "Score": f"{item['score'] * 100:.1f}%",
            }
        )
    return pd.DataFrame(rows)


def on_check(claim: str, history):
    claim = (claim or "").strip()
    history = history or []
    if not claim:
        return READY_HTML, "_No evidence yet._", history, history_to_dataframe(history)

    score, evidence_texts = run_fact_check(claim)
    verdict_html = format_verdict_html(score)
    evidence_md = format_evidence_markdown(evidence_texts)

    label, _ = score_to_verdict(score)
    history = history + [
        {
            "claim": claim,
            "score": score,
            "label": label,
            "evidence": evidence_texts,
        }
    ]
    return verdict_html, evidence_md, history, history_to_dataframe(history)


def on_clear_history():
    return [], history_to_dataframe([]), READY_HTML, "_No evidence yet._"


# --------------------------------------------------------------------------
# Static content
# --------------------------------------------------------------------------
SUMMARY_MD = """
### Wikipedia-grounded fact checking with RAG + BERT verifier.

This demo takes a short claim, retrieves the most relevant passages from an English Wikipedia
dump, and runs a BERT-based verifier over each `[claim, evidence]` pair. The per-pair logits
are maxpooled across the top-k retrieved chunks — one confident *supports* or *refutes* signal
is enough to drive the final decision — and the truthfulness score is computed as
`p(supports) / (p(supports) + p(refutes))`, ignoring the *not-enough-info* mass.

**Stack.** Embedder: `microsoft/harrier-oss-v1-0.6b` · Index: FAISS IVF4096 + PQ256
(~6.3 GB, 23.7M vectors) · Corpus: English Wikipedia, 200-word chunks with 50-word overlap,
title prepended as an extra signal · Verifier: BERT-family model over 5 retrieved chunks,
512-token context per pair.

_The score reflects how well the claim is supported by the retrieved Wikipedia evidence —
it's a grounded signal, not absolute truth. When evidence is weak the model leans toward
"uncertain"._
"""

BENCHMARK_DF = pd.DataFrame(
    [
        ["harrier-270m", "PQ160", "4.00 GB", "32.5%", "44.3%", "47.4%"],
        ["harrier-270m", "PQ320", "7.80 GB", "34.2%", "45.6%", "48.2%"],
        ["harrier-0.6b", "PQ128", "3.25 GB", "32.5%", "44.6%", "47.7%"],
        ["harrier-0.6b", "PQ256 ★", "6.29 GB", "35.5%", "47.3%", "50.0%"],
    ],
    columns=["Embedder", "Index", "Size", "Recall@1", "Recall@3", "Recall@5"],
)

QUICK_EXAMPLES = [
    "The Eiffel Tower was designed by Gustave Eiffel and is located in Paris, France.",
    "Mount Everest is the tallest mountain in the world and lies on the border between Nepal and China.",
]


# --------------------------------------------------------------------------
# UI
# --------------------------------------------------------------------------
CUSTOM_CSS = """
.gradio-container { max-width: 1200px !important; }

.hero {
    padding: 28px 32px;
    border-radius: 18px;
    background: linear-gradient(135deg, #eef2ff 0%, #f0f9ff 50%, #fef3c7 100%);
    border: 1px solid rgba(148, 163, 184, 0.25);
    margin-bottom: 16px;
}
.hero-tag {
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 12px;
    color: #475569;
    font-weight: 600;
}
.hero-title {
    font-size: 32px;
    font-weight: 700;
    margin: 6px 0 4px 0;
    color: #0f172a;
}
.hero-sub {
    color: #475569;
    font-size: 15px;
}

.section-tag {
    display: inline-block;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    font-size: 11px;
    font-weight: 600;
    color: #475569;
    background: #e2e8f0;
    padding: 4px 10px;
    border-radius: 999px;
    margin-bottom: 8px;
}

.verdict-card {
    padding: 20px;
    border-radius: 14px;
    background: #f8fafc;
    border: 1px solid rgba(148, 163, 184, 0.3);
    text-align: center;
    min-height: 160px;
}
.verdict-label {
    font-size: 13px;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}
.verdict-score {
    font-size: 40px;
    font-weight: 700;
    color: #0f172a;
    margin: 4px 0;
}
.verdict-caption {
    font-size: 12px;
    color: #64748b;
    margin-top: 6px;
}
"""

with gr.Blocks(theme=gr.themes.Soft(), css=CUSTOM_CSS, title="Fact Checker") as demo:
    history_state = gr.State([])

    # Hero
    gr.HTML(
        """
        <div class="hero">
            <div class="hero-tag">RAG Fact Checker</div>
            <div class="hero-title">Fact Checker</div>
            <div class="hero-sub">Paste a claim and get a truthfulness score grounded in Wikipedia evidence.</div>
        </div>
        """
    )

    # Project summary + benchmark
    with gr.Group():
        gr.HTML('<div class="section-tag">Project Summary</div>')
        gr.Markdown(SUMMARY_MD)
        with gr.Accordion("Retrieval benchmark (FEVER, 45 033 claims)", open=False):
            gr.Dataframe(value=BENCHMARK_DF, interactive=False, wrap=True)
            gr.Markdown(
                "_The ★ row is the configuration used in this demo. PQ256 on the 0.6b "
                "embedder gives the best recall/size trade-off that still fits in RAM._"
            )

    gr.Markdown("")  # spacer

    # Working area
    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML('<div class="section-tag">Claim</div>')
            claim_box = gr.Textbox(
                placeholder="e.g. The Eiffel Tower is in Paris",
                lines=5,
                show_label=False,
            )
            with gr.Row():
                check_btn = gr.Button("Check", variant="primary", scale=2)
                clear_btn = gr.Button("Clear history", scale=1)

            gr.HTML('<div class="section-tag" style="margin-top:12px;">Quick examples</div>')
            with gr.Row():
                ex_buttons = []
                for ex in QUICK_EXAMPLES:
                    preview = ex if len(ex) <= 60 else ex[:57] + "..."
                    ex_buttons.append(gr.Button(preview, size="sm"))

        with gr.Column(scale=1):
            gr.HTML('<div class="section-tag">Verdict</div>')
            verdict_out = gr.HTML(value=READY_HTML)

    # Evidence
    with gr.Group():
        gr.HTML('<div class="section-tag">Retrieved evidence</div>')
        evidence_out = gr.Markdown("_No evidence yet._")

    # Session log
    with gr.Group():
        gr.HTML('<div class="section-tag">Session log</div>')
        gr.Markdown("Every submission is kept for the duration of your session.")
        history_table = gr.Dataframe(
            value=history_to_dataframe([]),
            interactive=False,
            wrap=True,
        )

    # Wiring
    check_btn.click(
        fn=on_check,
        inputs=[claim_box, history_state],
        outputs=[verdict_out, evidence_out, history_state, history_table],
    )
    clear_btn.click(
        fn=on_clear_history,
        inputs=None,
        outputs=[history_state, history_table, verdict_out, evidence_out],
    )
    for btn, ex in zip(ex_buttons, QUICK_EXAMPLES):
        btn.click(fn=lambda ex=ex: ex, inputs=None, outputs=claim_box)


if __name__ == "__main__":
    demo.launch()
