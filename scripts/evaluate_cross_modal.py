# scripts/evaluate_cross_modal.py

import os
import json
import numpy as np
import torch
import torch.nn.functional as F
import faiss
from tqdm import tqdm

from src.snn_model import SNNEncoder

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(ROOT, "data", "processed")

CAP_SPIKE_DIR = os.path.join(DATA, "spikes", "captions")
FAISS_INDEX_PATH = os.path.join(DATA, "faiss_index.bin")
KEYS_PATH = os.path.join(DATA, "final_keys.json")
MODEL_PATH = os.path.join(DATA, "snn_encoder.pt")  # if saved

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EMB_DIM = 512

def load_index():
    return faiss.read_index(FAISS_INDEX_PATH)

def load_keys():
    with open(KEYS_PATH, "r") as f:
        return json.load(f)

def load_encoder(input_dim):
    model = SNNEncoder(
        in_dim=input_dim,
        hidden_dims=(256, 128),
        embedding_dim=EMB_DIM,
        tau_mem=20.0,
        dropout=0.05
    ).to(DEVICE)

    model.eval()
    return model

def main():
    print("Loading FAISS index...")
    index = load_index()
    keys = load_keys()
    key_to_id = {k: i for i, k in enumerate(keys)}

    recall1 = recall5 = mrr = 0
    valid = 0

    encoder = None

    print("Evaluating cross-modal retrieval (TEXT → IMAGE)...")

    for fname in tqdm(os.listdir(CAP_SPIKE_DIR)):
        if not fname.endswith(".npy"):
            continue

        query_key = fname.replace(".npy", "")
        if query_key not in key_to_id:
            continue

        spike = np.load(os.path.join(CAP_SPIKE_DIR, fname))
        spike = torch.tensor(spike, dtype=torch.float32).unsqueeze(0).to(DEVICE)

        if encoder is None:
            encoder = load_encoder(spike.shape[-1])

        with torch.no_grad():
            emb = encoder(spike)
            emb = F.normalize(emb, p=2, dim=1)
            emb = emb.cpu().numpy().astype("float32")

        scores, ids = index.search(emb, 5)
        retrieved = [keys[i] for i in ids[0]]

        valid += 1

        if query_key == retrieved[0]:
            recall1 += 1
        if query_key in retrieved:
            recall5 += 1
            rank = retrieved.index(query_key) + 1
            mrr += 1.0 / rank

    print("\n===== CROSS-MODAL RESULTS =====")
    print(f"Valid queries : {valid}")
    print(f"Recall@1      : {recall1 / valid:.4f}")
    print(f"Recall@5      : {recall5 / valid:.4f}")
    print(f"MRR           : {mrr / valid:.4f}")
    print("==============================")

if __name__ == "__main__":
    main()
