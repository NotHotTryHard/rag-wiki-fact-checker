"""Quick sanity check: count NaN/Inf vectors per shard."""

import os, glob
import numpy as np
from safetensors.numpy import load_file
from rag_config import EMBEDDINGS_DIR

shard_files = sorted(glob.glob(os.path.join(EMBEDDINGS_DIR, "shard_*.safetensors")))
total, total_bad = 0, 0

for path in shard_files:
    vecs = load_file(path)["vecs"]
    bad = ~np.isfinite(vecs).all(axis=1)
    n_bad = bad.sum()
    total += len(vecs)
    total_bad += n_bad
    if n_bad:
        print(f"{os.path.basename(path)}: {n_bad:,}/{len(vecs):,} bad ({100*n_bad/len(vecs):.1f}%)")
    else:
        print(f"{os.path.basename(path)}: OK ({len(vecs):,} vectors)")

print(f"\nTotal: {total_bad:,}/{total:,} bad ({100*total_bad/total:.1f}%)")
