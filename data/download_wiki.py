"""Download English Wikipedia and save to disk."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))

from datasets import load_dataset
from rag_config import DATASET_DIR

os.makedirs(os.path.dirname(DATASET_DIR), exist_ok=True)

if os.path.exists(DATASET_DIR):
    print(f"Already exists: {DATASET_DIR}, skipping.")
else:
    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    ds.save_to_disk(DATASET_DIR)
    print(f"Saved {len(ds):,} articles to {DATASET_DIR}")
