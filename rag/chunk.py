"""Chunk Wikipedia articles into overlapping text chunks and store in SQLite."""

import os
import sqlite3
from datasets import load_from_disk
from tqdm import tqdm
from config import DATASET_DIR, DB_PATH, CHUNK_SIZE, OVERLAP

ds = load_from_disk(DATASET_DIR)

os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
conn = sqlite3.connect(DB_PATH)
conn.execute("""
    CREATE TABLE IF NOT EXISTS chunks (
        id    INTEGER PRIMARY KEY,
        title TEXT,
        chunk TEXT
    )
""")
conn.execute("CREATE INDEX IF NOT EXISTS idx_title ON chunks(title)")

row = conn.execute("SELECT MAX(id) FROM chunks").fetchone()
chunk_id = (row[0] + 1) if row[0] is not None else 0

batch = []

for article in tqdm(ds, desc="Chunking"):
    words = article["text"].split()
    title = article["title"]

    for start in range(0, len(words), CHUNK_SIZE - OVERLAP):
        chunk_text = " ".join(words[start : start + CHUNK_SIZE])
        if len(chunk_text.strip()) < 50:
            continue
        batch.append((chunk_id, title, chunk_text))
        chunk_id += 1

        if len(batch) >= 10_000:
            conn.executemany("INSERT OR IGNORE INTO chunks VALUES (?, ?, ?)", batch)
            conn.commit()
            batch = []

if batch:
    conn.executemany("INSERT OR IGNORE INTO chunks VALUES (?, ?, ?)", batch)
    conn.commit()

total = conn.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]
print(f"Total chunks in DB: {total:,}")
conn.close()
