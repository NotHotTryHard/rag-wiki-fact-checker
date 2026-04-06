"""Download FEVER dataset and save to disk."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "rag"))

from datasets import load_dataset, DatasetDict
from rag_config import DATA_DIR

fever_dir = os.path.join(DATA_DIR, "datasets", "fever")

if os.path.exists(fever_dir):
    print(f"Already exists: {fever_dir}, skipping.")
else:
    print("Downloading fever/fever")
    ds = DatasetDict({
        "train": load_dataset("fever/fever", split="train"),
        "validation": load_dataset("fever/fever", split="validation"),
        "test": load_dataset("fever/fever", split="test"),
    })
    ds.save_to_disk(fever_dir)
    print(f"Saved {len(ds):,} examples to {fever_dir}")
