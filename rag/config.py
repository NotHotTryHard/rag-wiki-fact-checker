import os
import argparse

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, "data")
DATASET_DIR = os.path.join(DATA_DIR, "datasets", "wiki_en")
DB_PATH = os.path.join(DATA_DIR, "chunks", "wiki_en.db")

CHUNK_SIZE = 200
OVERLAP = 50

EMBEDDER = "microsoft/harrier-oss-v1-0.6b"
QUERY_PROMPT = "web_search_query"

EMBEDDER_TAG = EMBEDDER.rsplit("/", 1)[-1]
INDICES_DIR = os.path.join(DATA_DIR, "indices", EMBEDDER_TAG)

TRAIN_SIZE = 100000
ENCODE_BATCH = 256

PQ_VARIANTS = {
    "PQ64": "IVF4096,PQ64",
    "PQ128": "IVF4096,PQ128",
    "PQ256": "IVF4096,PQ256",
}

NPROBE = 32


def faiss_path(name):
    return os.path.join(INDICES_DIR, f"{name}.faiss")


def checkpoint_path(name):
    return os.path.join(INDICES_DIR, f".checkpoint_{name}")


def parse_gpu_args(extra_args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=None, help="GPU index")
    if extra_args:
        for arg in extra_args:
            parser.add_argument(*arg[0], **arg[1])
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    import torch
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    return args, device
