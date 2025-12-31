"""
Evaluate Retrieval Performance (Self-Retrieval)
-----------------------------------------------
Metrics:
- Recall@1
- Recall@5
- Mean Reciprocal Rank (MRR)

Definition of correctness:
- Query is correct if it retrieves itself
"""

import os
import sys
import json
import numpy as np
import faiss
from tqdm import tqdm

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

DATA_DIR = os.path.join(ROOT, "data", "processed")
EMB_PATH = os.path.join(DATA_DIR, "final_embeddings.npy")
KEYS_PATH = os.path.join(DATA_DIR, "final_keys.json")
FAISS_INDEX_PATH = os.path.join(DATA_DIR, "faiss_index.bin")

TOP_K = 5

def main():
    print("Loading embeddings and index...")

    embeddings = np.load(EMB_PATH).astype("float32")
    with open(KEYS_PATH, "r") as f:
        keys = json.load(f)

    index = faiss.read_index(FAISS_INDEX_PATH)

    recall1 = 0
    recall5 = 0
    mrr = 0

    for i in tqdm(range(len(embeddings)), desc="Evaluating"):
        query = embeddings[i].reshape(1, -1)
        query_key = keys[i]

        _, ids = index.search(query, TOP_K)
        retrieved_keys = [keys[j] for j in ids[0]]

        # Recall@1
        if retrieved_keys[0] == query_key:
            recall1 += 1

        # Recall@5
        if query_key in retrieved_keys:
            recall5 += 1

        # MRR
        for rank, key in enumerate(retrieved_keys):
            if key == query_key:
                mrr += 1.0 / (rank + 1)
                break

    n = len(embeddings)
    print("\n===== RETRIEVAL RESULTS =====")
    print(f"Recall@1 : {recall1 / n:.4f}")
    print(f"Recall@5 : {recall5 / n:.4f}")
    print(f"MRR      : {mrr / n:.4f}")
    print("============================")

if __name__ == "__main__":
    main()
