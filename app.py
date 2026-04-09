import os
import sys
import sqlite3
import html
from typing import Dict, List, Tuple

import gradio as gr


REPO_URL = "https://github.com/NotHotTryHard/rag-wiki-fact-checker"
HF_DATASET_REPO = "NotHotTryHard/fact-checker-data"
TOP_K = 5
GITHUB_ICON_PATH = (
    "M8 0C3.58 0 0 3.58 0 8a8 8 0 0 0 5.47 7.59c.4.07.55-.17.55-.38"
    " 0-.19-.01-.82-.01-1.49-2.01.37-2.53-.49-2.69-.94-.09-.23-.48-.94"
    "-.82-1.13-.28-.15-.68-.52-.01-.53.63-.01 1.08.58 1.23.82.72 1.21"
    " 1.87.87 2.33.66.07-.52.28-.87.5-1.07-1.78-.2-3.64-.89-3.64-3.95"
    " 0-.87.31-1.59.82-2.15-.08-.2-.36-1.02.08-2.12 0 0 .67-.21 2.2.82"
    "a7.6 7.6 0 0 1 4 0c1.53-1.04 2.2-.82 2.2-.82.44 1.1.16 1.92.08"
    " 2.12.51.56.82 1.27.82 2.15 0 3.07-1.87 3.75-3.65 3.95.29.25.54.73"
    " .54 1.48 0 1.07-.01 1.93-.01 2.2 0 .21.15.46.55.38A8 8 0 0 0 16 8"
    "c0-4.42-3.58-8-8-8Z"
)

RUNTIME: Dict[str, object] = {
    "loaded": False,
    "embedder": None,
    "tokenizer": None,
    "sayer": None,
    "index": None,
    "conn": None,
    "query_prompt": None,
    "max_length": None,
    "top_k": TOP_K,
}

EXAMPLE_CLAIMS = [
    'Trump was on Epstein island',
    "The Eiffel Tower was designed by Gustave Eiffel and is located in Paris, France.",
    "The Eiffel Tower was designed by Gustavo Fring and is located in Berlin, Germany.",
]

THEME = gr.themes.Soft(
    primary_hue="emerald",
    secondary_hue="amber",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Space Grotesk"), "ui-sans-serif", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "monospace"],
)

HERO_HTML = f"""
<section id="hero-panel">
  <div class="hero-topline">
    <div class="hero-kicker">Wikipedia grounded verifier</div>
    <a class="github-link" href="{REPO_URL}" target="_blank" rel="noreferrer">
      <svg viewBox="0 0 16 16" aria-hidden="true">
        <path fill="currentColor" d="{GITHUB_ICON_PATH}"/>
      </svg>
      <span>GitHub repo</span>
    </a>
  </div>
  <div class="hero-grid">
    <div class="hero-copy">
      <h1>RAG Wiki Fact Checker</h1>
      <p>Paste a claim and inspect verdict confidence with retrieved evidence snippets.</p>
    </div>
    <div class="hero-chip-row">
      <span class="hero-chip">FAISS IVF + PQ</span>
      <span class="hero-chip">Wikipedia chunks</span>
      <span class="hero-chip">HF</span>
    </div>
  </div>
</section>
"""

PROJECT_SUMMARY_HTML = """
<div class="workspace-intro project-summary-card">
  <div class="section-heading">
    <div class="section-kicker">Project Summary</div>
    <h2>Retrieval + verifier pipeline for factuality checks.</h2>
    <p class="project-summary-copy">
      This demo takes a short claim, retrieves relevant passages from an English
      Wikipedia dump, and runs a BERT-based verifier over each
      <code>[claim, evidence]</code> pair. Per-pair logits are maxpooled across top-k chunks:
      one strong supports/refutes signal is enough to drive the final decision.
    </p>
    <div class="formula-line">
      <span class="formula-key">Truthfulness</span>
      <span class="formula-eq">=</span>
      <span class="formula-frac">
        <span class="formula-num">p(supports)</span>
        <span class="formula-den">p(supports) + p(refutes)</span>
      </span>
    </div>
    <p class="project-summary-copy">
      The not-enough-info mass is excluded from this score.
    </p>
    <p class="project-summary-copy">
      Stack: <code>microsoft/harrier-oss-v1-0.6b</code> embeddings,
      FAISS IVF4096 + PQ256 index (~6.3 GB, 23.7M vectors), English Wikipedia
      chunks (200 words, overlap 50, title-prepended), and a BERT-family verifier
      over 5 retrieved chunks with 512-token context per pair.
    </p>
  </div>
  <div class="project-metrics-shell">
    <div class="project-metrics-label">Embedding Benchmark Snapshot</div>
    <div class="project-metrics-table-wrap">
      <table class="project-metrics-table" aria-label="Embedding benchmark">
        <thead>
          <tr>
            <th scope="col">Embedder / Index</th>
            <th scope="col">Recall@1</th>
            <th scope="col">Recall@5</th>
          </tr>
        </thead>
        <tbody>
          <tr>
            <td data-label="Embedder / Index">harrier-270m + PQ160 (4.00 GB)</td>
            <td data-label="Recall@1">32.5%</td>
            <td data-label="Recall@5">47.4%</td>
          </tr>
          <tr>
            <td data-label="Embedder / Index">harrier-270m + PQ320 (7.80 GB)</td>
            <td data-label="Recall@1">34.2%</td>
            <td data-label="Recall@5">48.2%</td>
          </tr>
          <tr>
            <td data-label="Embedder / Index">harrier-0.6b + PQ128 (3.25 GB)</td>
            <td data-label="Recall@1">32.5%</td>
            <td data-label="Recall@5">47.7%</td>
          </tr>
          <tr>
            <td data-label="Embedder / Index">harrier-0.6b + PQ256 (6.29 GB)</td>
            <td data-label="Recall@1">35.5%</td>
            <td data-label="Recall@5">50.0%</td>
          </tr>
        </tbody>
      </table>
    </div>
    <div class="project-metrics-cards" aria-hidden="true">
      <div class="project-metrics-mobile-card">
        <div class="project-metrics-mobile-model">harrier-270m + PQ160</div>
        <div class="project-metrics-mobile-stat"><span>Recall@1</span><strong>32.5%</strong></div>
        <div class="project-metrics-mobile-stat"><span>Recall@5</span><strong>47.4%</strong></div>
      </div>
      <div class="project-metrics-mobile-card">
        <div class="project-metrics-mobile-model">harrier-270m + PQ320</div>
        <div class="project-metrics-mobile-stat"><span>Recall@1</span><strong>34.2%</strong></div>
        <div class="project-metrics-mobile-stat"><span>Recall@5</span><strong>48.2%</strong></div>
      </div>
      <div class="project-metrics-mobile-card">
        <div class="project-metrics-mobile-model">harrier-0.6b + PQ256</div>
        <div class="project-metrics-mobile-stat"><span>Recall@1</span><strong>35.5%</strong></div>
        <div class="project-metrics-mobile-stat"><span>Recall@5</span><strong>50.0%</strong></div>
      </div>
    </div>
    <p class="project-summary-note">
      Score reflects support in retrieved evidence, not absolute truth.
      При слабом retrieval система чаще выдает uncertain.
    </p>
  </div>
</div>
"""

CSS = """
:root {
  color-scheme: light;
  --paper-ink: #132238;
  --paper-muted: #5b6b7d;
  --paper-soft: #708295;
  --paper-line: rgba(19, 34, 56, 0.12);
  --paper-panel: rgba(255, 255, 255, 0.9);
  --paper-panel-strong: rgba(255, 255, 255, 0.97);
  --paper-shadow: 0 28px 90px rgba(15, 23, 42, 0.12);
  --paper-blue: #1d4ed8;
  --paper-teal: #0f766e;
}

.gradio-container {
  background:
    radial-gradient(circle at 12% 12%, rgba(15, 118, 110, 0.18), transparent 24%),
    radial-gradient(circle at 88% 16%, rgba(37, 99, 235, 0.18), transparent 26%),
    radial-gradient(circle at 82% 78%, rgba(245, 158, 11, 0.16), transparent 24%),
    linear-gradient(180deg, #f6f7f1 0%, #eef3f8 58%, #f7efe3 100%);
  color: var(--paper-ink);
}

.gradio-container::before {
  content: "";
  position: fixed;
  inset: 0;
  pointer-events: none;
  background:
    linear-gradient(125deg, rgba(255, 255, 255, 0.18), transparent 28%),
    linear-gradient(305deg, rgba(255, 255, 255, 0.10), transparent 26%);
  opacity: 0.6;
  animation: page-float 18s ease-in-out infinite;
}

@keyframes page-float {
  0%, 100% { transform: translate3d(0, 0, 0); }
  50% { transform: translate3d(0, -10px, 0); }
}

#app-shell {
  max-width: 1240px;
  margin: 0 auto;
  padding: 22px 18px 44px;
}

.panel {
  position: relative;
  overflow: hidden;
  padding: 22px;
  border: 1px solid rgba(68, 64, 60, 0.18);
  border-radius: 28px;
  background: linear-gradient(180deg, var(--paper-panel-strong), var(--paper-panel));
  box-shadow: var(--paper-shadow);
  backdrop-filter: blur(18px);
  transition: transform 0.25s ease, box-shadow 0.25s ease, border-color 0.25s ease;
}

.panel::before {
  content: "";
  position: absolute;
  inset: 0;
  pointer-events: none;
  background: linear-gradient(145deg, rgba(255, 255, 255, 0.55), transparent 34%);
}

#input-panel,
#result-panel {
  border-radius: 28px !important;
  overflow: hidden;
}

#output-panel {
  border-radius: 28px !important;
  overflow: hidden;
}

#input-panel > div,
#result-panel > div,
#output-panel > div {
  border-radius: 24px !important;
  overflow: hidden !important;
}

#input-panel .gr-group,
#result-panel .gr-group,
#input-panel .block,
#result-panel .block,
#input-panel .form,
#result-panel .form,
#input-panel .gr-box,
#result-panel .gr-box,
#output-panel .gr-group,
#output-panel .block,
#output-panel .form,
#output-panel .gr-box,
#output-panel .gr-html,
#output-panel .gr-markdown {
  border-radius: 22px !important;
  overflow: hidden !important;
}

#hero-panel {
  position: relative;
  overflow: hidden;
  margin-bottom: 14px;
  padding: 18px 22px;
  border-radius: 30px;
  background:
    radial-gradient(circle at top right, rgba(255, 255, 255, 0.84), transparent 34%),
    linear-gradient(135deg, rgba(15, 118, 110, 0.12) 0%, rgba(37, 99, 235, 0.16) 52%, rgba(245, 158, 11, 0.18) 100%),
    rgba(255, 255, 255, 0.95);
  border: 1px solid rgba(19, 34, 56, 0.10);
  box-shadow: 0 24px 70px rgba(37, 99, 235, 0.14);
}

.hero-topline {
  position: relative;
  z-index: 1;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 16px;
  margin-bottom: 10px;
}

.hero-kicker {
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.13em;
  text-transform: uppercase;
  color: rgba(19, 34, 56, 0.78);
}

.hero-copy h1 {
  margin: 0;
  font-size: clamp(2.3rem, 4vw, 3.4rem);
  line-height: 1;
  letter-spacing: -0.07em;
}

.hero-copy p {
  margin: 8px 0 0;
  max-width: 34rem;
  color: #556476;
  font-size: 0.98rem;
  line-height: 1.45;
}

.github-link {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  padding: 10px 14px;
  border-radius: 999px;
  border: 1px solid rgba(19, 34, 56, 0.10);
  background: rgba(255, 255, 255, 0.74);
  color: var(--paper-ink);
  text-decoration: none;
  font-size: 0.92rem;
  font-weight: 700;
}

.github-link svg {
  width: 16px;
  height: 16px;
}

.hero-chip-row {
  display: flex;
  flex-wrap: wrap;
  gap: 10px;
  margin-top: 18px;
}

.hero-chip {
  padding: 9px 13px;
  border-radius: 999px;
  background: rgba(255, 255, 255, 0.62);
  border: 1px solid rgba(19, 34, 56, 0.09);
  font-size: 0.9rem;
}

.section-heading {
  margin-bottom: 16px;
}

.section-kicker {
  display: inline-flex;
  align-items: center;
  margin-bottom: 10px;
  padding: 7px 12px;
  border-radius: 999px;
  background: rgba(15, 118, 110, 0.10);
  color: var(--paper-teal);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.paper-input-shell {
  position: relative;
  z-index: 1;
  padding: 14px;
  border-radius: 28px;
  overflow: hidden;
  border: 1px solid rgba(15, 118, 110, 0.12);
  background: linear-gradient(180deg, rgba(255, 255, 255, 0.98), rgba(244, 248, 252, 0.94));
}

#claim-input,
#claim-input > div,
#claim-input .form,
#claim-input .wrap,
#claim-input .scroll-hide {
  border-radius: 24px !important;
}

.paper-input textarea,
#claim-input textarea,
.paper-input textarea:focus,
#claim-input textarea:focus {
  min-height: 120px !important;
  border-radius: 24px !important;
}

.example-menu {
  display: grid;
  gap: 10px;
}

.example-btn button {
  width: 100%;
  min-height: 44px;
  border-radius: 20px !important;
  justify-content: flex-start;
  text-align: left;
  white-space: normal !important;
  line-height: 1.3;
}

.evidence-stack {
  display: grid;
  gap: 10px;
  margin-top: 10px;
}

.evidence-item {
  border: 1px solid var(--paper-line);
  border-radius: 14px;
  background: rgba(255, 255, 255, 0.78);
  padding: 8px 12px;
}

.evidence-item summary {
  cursor: pointer;
  font-weight: 600;
  color: #0f172a;
  outline: none;
}

.evidence-item-body {
  margin-top: 8px;
  color: #334155;
  line-height: 1.55;
}

.project-summary-card {
  position: relative;
  overflow: hidden;
  padding: 24px;
  border: 1px solid rgba(19, 34, 56, 0.12);
  border-radius: 30px;
  background:
    radial-gradient(circle at top right, rgba(255, 255, 255, 0.4), transparent 30%),
    linear-gradient(145deg, rgba(37, 99, 235, 0.08), rgba(15, 118, 110, 0.06));
  box-shadow: 0 24px 60px rgba(15, 23, 42, 0.08);
}

.project-summary-copy,
.project-summary-note {
  color: var(--paper-muted);
}

.formula-line {
  margin: 8px auto 12px;
  text-align: center;
  color: var(--paper-ink);
  font-size: 1.02rem;
  line-height: 1.4;
}

.formula-key {
  font-weight: 700;
}

.formula-eq {
  margin: 0 8px;
}

.formula-frac {
  display: inline-grid;
  grid-template-rows: auto auto;
  text-align: center;
  vertical-align: middle;
  min-width: 260px;
}

.formula-num {
  border-bottom: 1px solid rgba(19, 34, 56, 0.6);
  padding: 0 8px 2px;
}

.formula-den {
  padding: 2px 8px 0;
}

.project-metrics-shell {
  margin-top: 18px;
  margin-left: auto;
  margin-right: auto;
  max-width: 820px;
  padding: 16px;
  border: 1px solid rgba(19, 34, 56, 0.1);
  border-radius: 22px;
  background: rgba(255, 255, 255, 0.82);
}

.project-metrics-label {
  margin-bottom: 10px;
  color: var(--paper-blue);
  font-size: 0.78rem;
  font-weight: 700;
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

.project-metrics-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  border: 0;
  border-radius: 0;
  background: rgba(255, 255, 255, 0.9);
}

.project-metrics-table-wrap {
  border: 1px solid rgba(19, 34, 56, 0.08);
  overflow: hidden;
  overflow-x: auto;
  overflow-y: hidden;
  border-radius: 18px;
  background: rgba(255, 255, 255, 0.9);
  max-width: 780px;
  margin: 0 auto;
}

.project-metrics-cards {
  display: none;
}

.project-metrics-table th,
.project-metrics-table td {
  padding: 12px 14px;
  text-align: left;
  border-bottom: 1px solid rgba(19, 34, 56, 0.08);
}

.project-metrics-table th + th,
.project-metrics-table td + td {
  border-left: 1px solid rgba(19, 34, 56, 0.08);
}

.project-metrics-table th {
  background: rgba(15, 118, 110, 0.08);
  color: var(--paper-muted);
  font-size: 0.76rem;
  font-weight: 700;
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

.project-metrics-table tbody tr:last-child td {
  border-bottom: none;
}

.project-metrics-table td:nth-child(2),
.project-metrics-table td:nth-child(3),
.project-metrics-table th:nth-child(2),
.project-metrics-table th:nth-child(3) {
  text-align: right;
  width: 18%;
}

.verdict-card {
  border-radius: 24px;
  border: 1px solid var(--paper-line);
  padding: 14px;
  background: rgba(255, 255, 255, 0.86);
}

.verdict-row {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 168px;
  gap: 12px;
  align-items: stretch;
}

.verdict-main {
  min-width: 0;
}

.verdict-side {
  border-radius: 20px;
  display: grid;
  align-content: center;
  justify-items: end;
  text-align: right;
}

.verdict-title {
  font-size: 1.05rem;
  font-weight: 700;
}

.verdict-metrics-row {
  margin-top: 10px;
}

.truth-slider {
  display: grid;
  gap: 6px;
}

.truth-slider-labels {
  display: flex;
  justify-content: space-between;
  font-size: 0.72rem;
  letter-spacing: 0.06em;
  font-weight: 700;
  color: var(--paper-soft);
}

.truth-slider-track {
  position: relative;
  height: 12px;
  border-radius: 999px;
  background: linear-gradient(90deg, #dc2626 0%, #f59e0b 45%, #16a34a 100%);
  box-shadow: inset 0 0 0 1px rgba(19, 34, 56, 0.12);
}

.truth-slider-thumb {
  position: absolute;
  top: 50%;
  transform: translate(-50%, -50%);
  width: 16px;
  height: 16px;
  border-radius: 999px;
  border: 2px solid #ffffff;
  box-shadow: 0 2px 10px rgba(2, 6, 23, 0.28);
  background: #0f172a;
}

.groundedness-label {
  font-size: 0.68rem;
  font-weight: 700;
  letter-spacing: 0.07em;
  text-transform: uppercase;
  color: var(--paper-soft);
}

.groundedness-value {
  margin-top: 2px;
  font-size: 1rem;
  font-weight: 800;
  color: var(--paper-ink);
}

.workspace-row {
  gap: 16px;
}

@media (max-width: 960px) {
  #app-shell {
    padding: 18px 12px 42px;
  }
  .hero-topline {
    align-items: flex-start;
    flex-direction: column;
  }
}

@media (max-width: 640px) {
  .panel {
    padding: 18px;
    border-radius: 24px;
  }
  .project-summary-card {
    padding: 18px;
    border-radius: 24px;
  }
  .verdict-row {
    grid-template-columns: minmax(0, 1fr);
  }
  .verdict-side {
    justify-items: start;
    text-align: left;
  }
  .project-metrics-shell {
    padding: 12px;
  }
  .project-metrics-table-wrap {
    display: none;
  }
  .project-metrics-cards {
    display: grid;
    gap: 10px;
  }
  .project-metrics-mobile-card {
    display: grid;
    gap: 10px;
    padding: 12px;
    border: 1px solid rgba(19, 34, 56, 0.08);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.92);
    box-shadow: 0 10px 24px rgba(15, 23, 42, 0.06);
  }
  .project-metrics-mobile-model {
    font-size: 0.98rem;
    font-weight: 700;
    line-height: 1.35;
  }
  .project-metrics-mobile-stat {
    display: grid;
    grid-template-columns: minmax(0, 1fr) auto;
    gap: 10px;
    padding-top: 8px;
    border-top: 1px solid rgba(19, 34, 56, 0.08);
    font-size: 0.9rem;
    align-items: center;
  }
  .project-metrics-mobile-stat span {
    color: var(--paper-soft);
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
  }
}
"""


def score_to_verdict(score: float) -> Tuple[str, str]:
    if score >= 0.75:
        return "Likely TRUE", "#16a34a"
    if score <= 0.25:
        return "Likely FALSE", "#dc2626"
    return "Uncertain", "#d97706"


def _load_runtime() -> Dict[str, object]:
    if RUNTIME["loaded"]:
        return RUNTIME

    base_dir = os.path.dirname(__file__)
    sys.path.append(os.path.join(base_dir, "rag"))
    sys.path.append(os.path.join(base_dir, "model"))

    from rag_config import EMBEDDER, QUERY_PROMPT, NPROBE
    from model_config import MODEL_NAME, MAX_LENGTH, NUM_LABELS, TOP_K as MODEL_TOP_K
    from model import TruthfulnessSayer

    import numpy as np
    import faiss
    import torch
    from transformers import AutoTokenizer
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import hf_hub_download

    weights = hf_hub_download(HF_DATASET_REPO, "best.pt", repo_type="dataset")
    index_path = hf_hub_download(HF_DATASET_REPO, "PQ256.faiss", repo_type="dataset")
    db_path = hf_hub_download(HF_DATASET_REPO, "wiki_en.db", repo_type="dataset")

    embedder = SentenceTransformer(EMBEDDER)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    sayer = TruthfulnessSayer(MODEL_NAME, NUM_LABELS, weights)

    index = faiss.read_index(index_path)
    faiss.extract_index_ivf(index).nprobe = NPROBE
    conn = sqlite3.connect(db_path, check_same_thread=False)

    RUNTIME.update(
        {
            "loaded": True,
            "embedder": embedder,
            "tokenizer": tokenizer,
            "sayer": sayer,
            "index": index,
            "conn": conn,
            "query_prompt": QUERY_PROMPT,
            "max_length": MAX_LENGTH,
            "top_k": MODEL_TOP_K,
            "np": np,
            "torch": torch,
        }
    )
    return RUNTIME


def _run_real_fact_check(claim: str) -> Tuple[float, float, List[str]]:
    rt = _load_runtime()

    np = rt["np"]
    torch = rt["torch"]
    query_emb = rt["embedder"].encode(
        [claim], prompt_name=rt["query_prompt"], normalize_embeddings=True
    ).astype(np.float32)

    _, ids_found = rt["index"].search(query_emb, rt["top_k"])
    chunk_ids = [i for i in ids_found[0].tolist() if i >= 0]

    evidence_texts: List[str] = []
    if chunk_ids:
        placeholders = ",".join("?" * len(chunk_ids))
        rows = rt["conn"].execute(
            f"SELECT id, chunk FROM chunks WHERE id IN ({placeholders})", chunk_ids
        ).fetchall()
        id_to_text = {row[0]: row[1] for row in rows}
        evidence_texts = [id_to_text.get(i, "") for i in chunk_ids]

    while len(evidence_texts) < rt["top_k"]:
        evidence_texts.append("")

    pairs = rt["tokenizer"](
        [claim] * rt["top_k"],
        evidence_texts,
        truncation=True,
        max_length=rt["max_length"],
        padding="longest",
        return_tensors="pt",
    )
    input_ids = pairs["input_ids"].unsqueeze(0)
    attention_mask = pairs["attention_mask"].unsqueeze(0)

    with torch.no_grad():
        truthfulness, groundedness, _, _, _ = rt["sayer"].predict_components(
            input_ids, attention_mask
        )

    return float(truthfulness.item()), float(groundedness.item()), evidence_texts


def _render_evidence_html(evidence: List[str]) -> str:
    items: List[str] = []
    for idx, text in enumerate(evidence, start=1):
        clean = (text or "").strip()
        if not clean:
            continue
        first_line = clean.splitlines()[0].strip()
        if len(first_line) > 140:
            first_line = first_line[:137] + "..."
        body = html.escape(clean).replace("\n", "<br>")
        items.append(
            "<details class='evidence-item'>"
            f"<summary>Chunk #{idx}: {html.escape(first_line)}</summary>"
            f"<div class='evidence-item-body'>{body}</div>"
            "</details>"
        )

    if not items:
        return (
            "<div class='evidence-stack'>"
            "<div class='evidence-item'><summary>No evidence retrieved.</summary></div>"
            "</div>"
        )

    return "<div class='evidence-stack'>" + "".join(items) + "</div>"


def run_demo(claim: str):
    claim = (claim or "").strip()
    if not claim:
        return (
            "<div class='verdict-card'><b>Waiting for input...</b></div>",
            "<div class='verdict-card verdict-side'></div>",
            _render_evidence_html([]),
        )

    score, groundedness, evidence = _run_real_fact_check(claim)

    label, color = score_to_verdict(score)
    slider_pos = max(0.0, min(100.0, score * 100.0))
    slider_html = (
        "<div class='verdict-card verdict-main'>"
        f"<div class='verdict-title' style='color:{color};'>{label}</div>"
        "<div class='truth-slider'>"
        "<div class='truth-slider-labels'><span>FALSE</span><span>TRUE</span></div>"
        "<div class='truth-slider-track'>"
        f"<div class='truth-slider-thumb' style='left:{slider_pos:.1f}%;'></div>"
        "</div>"
        f"<div style='font-size:0.82rem;color:#334155;'>Truthfulness: <b>{score:.3f}</b></div>"
        "</div>"
        "</div>"
    )
    g_pct = groundedness * 100
    if g_pct >= 70:
        g_label, g_color = "Strong", "#16a34a"
    elif g_pct >= 40:
        g_label, g_color = "Moderate", "#d97706"
    else:
        g_label, g_color = "Weak", "#dc2626"

    ground_html = (
        "<div class='verdict-card verdict-side' style='text-align:center;'>"
        f"<div style='font-size:0.7rem;text-transform:uppercase;letter-spacing:0.08em;"
        f"font-weight:600;color:{g_color};'>{g_label}</div>"
        f"<div style='font-size:2rem;font-weight:700;color:{g_color};margin:6px 0;'>{g_pct:.0f}%</div>"
        "<div style='font-size:0.75rem;color:#64748b;'>Groundedness</div>"
        "</div>"
    )

    return slider_html, ground_html, _render_evidence_html(evidence)

print("Loading models and index...")
_load_runtime()
print("Runtime ready.")


with gr.Blocks(title="RAG Wiki Fact Checker") as demo:
    with gr.Column(elem_id="app-shell"):
        gr.HTML(HERO_HTML)

        with gr.Row(equal_height=True, elem_classes=["workspace-row"]):
            with gr.Column(scale=8):
                with gr.Group(elem_id="input-panel", elem_classes=["panel"]):
                    gr.HTML(
                        """
                        <div class="section-heading">
                          <div class="section-kicker">Interactive Playground</div>
                        </div>
                        """
                    )
                    with gr.Column(elem_classes=["paper-input-shell"]):
                        claim = gr.Textbox(
                            label="Claim",
                            placeholder="Введите утверждение для проверки...",
                            lines=2,
                            elem_id="claim-input",
                            elem_classes=["paper-input"],
                        )
                    with gr.Row():
                        run_btn = gr.Button("Check claim", variant="primary")

            with gr.Column(scale=4):
                with gr.Group(elem_id="result-panel", elem_classes=["panel"]):
                    gr.HTML(
                        """
                        <div class="section-heading">
                          <div class="section-kicker">Examples</div>
                        </div>
                        """
                    )
                    with gr.Column(elem_classes=["example-menu"]):
                        ex1_btn = gr.Button(EXAMPLE_CLAIMS[0], elem_classes=["example-btn"])
                        ex2_btn = gr.Button(EXAMPLE_CLAIMS[1], elem_classes=["example-btn"])
                        ex3_btn = gr.Button(EXAMPLE_CLAIMS[2], elem_classes=["example-btn"])

        with gr.Row():
            with gr.Column(scale=12):
                with gr.Group(elem_id="output-panel", elem_classes=["panel"]):
                    gr.HTML(
                        """
                        <div class="section-heading">
                          <div class="section-kicker">Result</div>
                        </div>
                        """
                    )
                    with gr.Row():
                        with gr.Column(scale=3):
                            verdict_slider = gr.HTML("<div class='verdict-card'><b>Waiting for input...</b></div>")
                        with gr.Column(scale=1, min_width=120):
                            verdict_ground = gr.HTML("<div class='verdict-card verdict-side'></div>")
                    evidence = gr.HTML(_render_evidence_html([]))

        gr.HTML(PROJECT_SUMMARY_HTML)

    run_btn.click(
        fn=run_demo,
        inputs=[claim],
        outputs=[verdict_slider, verdict_ground, evidence],
    )
    claim.submit(
        fn=run_demo,
        inputs=[claim],
        outputs=[verdict_slider, verdict_ground, evidence],
    )
    ex1_btn.click(fn=lambda: EXAMPLE_CLAIMS[0], outputs=[claim])
    ex2_btn.click(fn=lambda: EXAMPLE_CLAIMS[1], outputs=[claim])
    ex3_btn.click(fn=lambda: EXAMPLE_CLAIMS[2], outputs=[claim])


if __name__ == "__main__":
    demo.launch(css=CSS, theme=THEME)
