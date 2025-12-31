"""
Robust SNN + FAISS Retrieval Script
---------------------------------
Supports:
- Image-to-image retrieval
- Text-to-image retrieval (prototype)
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn.functional as F
import faiss

# -------------------------------------------------
# Ensure project root is on PYTHONPATH
# -------------------------------------------------
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.snn_model import SNNEncoder

# -------------------------------------------------
# PATHS (MATCH YOUR FOLDER EXACTLY)
# -------------------------------------------------
DATA_DIR = os.path.join(ROOT, "data", "processed")

SPIKE_IMG_DIR = os.path.join(DATA_DIR, "spikes", "images")
SPIKE_CAP_DIR = os.path.join(DATA_DIR, "spikes", "captions")

FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")
KEYS_PATH = os.path.join(DATA_DIR, "final_keys.json")

# -------------------------------------------------
# CONFIG
# -------------------------------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
T = 60
EMB_DIM = 512

# -------------------------------------------------
# UTILITIES
# -------------------------------------------------

def load_index():
    if not os.path.exists(FAISS_INDEX_PATH):
        raise FileNotFoundError(f"FAISS index not found: {FAISS_INDEX_PATH}")
    return faiss.read_index(FAISS_INDEX_PATH)


def load_keys():
    if not os.path.exists(KEYS_PATH):
        raise FileNotFoundError(f"Key file not found: {KEYS_PATH}")
    with open(KEYS_PATH, "r") as f:
        return json.load(f)


def load_spike(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Spike file not found: {path}")
    spikes = np.load(path)
    spikes = torch.tensor(spikes, dtype=torch.float32).unsqueeze(0)
    return spikes.to(DEVICE)


# -------------------------------------------------
# ENCODER
# -------------------------------------------------

def load_encoder(in_dim):
    model = SNNEncoder(
        in_dim=in_dim,
        hidden_dims=(256, 128),
        embedding_dim=EMB_DIM,
        tau_mem=20.0,
        dropout=0.05,
    ).to(DEVICE)
    model.eval()
    return model


# -------------------------------------------------
# RETRIEVAL
# -------------------------------------------------

def retrieve(query_emb, index, keys, top_k=5):
    query_emb = query_emb.astype("float32")
    scores, ids = index.search(query_emb, top_k)

    results = []
    for rank, idx in enumerate(ids[0]):
        results.append((rank + 1, keys[idx], scores[0][rank]))
    return results


# -------------------------------------------------
# MAIN
# -------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="SNN + FAISS Retrieval")
    parser.add_argument("--image", type=str, help="Path to image (.jpg)")
    parser.add_argument("--text", type=str, help="Text query (prototype)")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    if not args.image and not args.text:
        parser.error("Provide --image or --text")

    print("Using device:", DEVICE)

    index = load_index()
    keys = load_keys()

    if args.image:
        base = os.path.basename(args.image)
        spike_path = os.path.join(SPIKE_IMG_DIR, base + ".npy")
        spikes = load_spike(spike_path)
        encoder = load_encoder(spikes.shape[-1])

    else:
        # Prototype text retrieval (uses existing caption spikes)
        ref_key = keys[0]
        spike_path = os.path.join(SPIKE_CAP_DIR, ref_key + ".npy")
        spikes = load_spike(spike_path)
        encoder = load_encoder(spikes.shape[-1])

    with torch.no_grad():
        emb = encoder(spikes)
        emb = F.normalize(emb, p=2, dim=1)
        emb = emb.cpu().numpy()

    results = retrieve(emb, index, keys, args.topk)

    print("\nTop Results:\n")
    for rank, key, score in results:
        print(f"{rank:02d}. {key}  | similarity={score:.4f}")


if __name__ == "__main__":
    main()
