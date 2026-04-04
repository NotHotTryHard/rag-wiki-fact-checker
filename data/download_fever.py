"""Download FEVER dataset and save to disk."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))

from datasets import load_dataset
from config import DATA_DIR

fever_dir = os.path.join(DATA_DIR, "datasets", "fever")

if os.path.exists(fever_dir):
    print(f"Already exists: {fever_dir}, skipping.")
else:
    print("Downloading fever/fever v1.0 paper_dev ...")
    ds = load_dataset("fever/fever", "v1.0", split="paper_dev")
    ds.save_to_disk(fever_dir)
    print(f"Saved {len(ds):,} examples to {fever_dir}")
