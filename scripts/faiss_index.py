# scripts/faiss_index.py
import os
import json
import numpy as np
import faiss

# ============================================================
# Paths
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROC_DIR = os.path.join(ROOT, "data", "processed")
EMB_PATH = os.path.join(PROC_DIR, "final_embeddings.npy")
KEYS_PATH = os.path.join(PROC_DIR, "final_keys.json")

OUT_INDEX = os.path.join(PROC_DIR, "faiss_index.bin")
OUT_IDMAP = os.path.join(PROC_DIR, "faiss_id_map.json")

# ============================================================
# Load Embeddings + Keys
# ============================================================

print("Loading embeddings...")
embeddings = np.load(EMB_PATH).astype("float32")
print("Embedding shape:", embeddings.shape)

with open(KEYS_PATH, "r") as f:
    keys = json.load(f)

assert len(keys) == embeddings.shape[0], "Key count mismatch with embeddings"


# ============================================================
# Normalize embeddings for cosine similarity
# ============================================================

faiss.normalize_L2(embeddings)

dim = embeddings.shape[1]
print(f"Embedding dimension: {dim}")

# ============================================================
# Build FAISS index (cosine → inner product)
# ============================================================

print("Building FAISS index...")

index = faiss.IndexFlatIP(dim)     # IP = cosine similarity because vectors are normalized
index.add(embeddings)

print("FAISS index built.")
print("Total vectors indexed:", index.ntotal)

# ============================================================
# Save index + mapping
# ============================================================

faiss.write_index(index, OUT_INDEX)
print("Saved index →", OUT_INDEX)

with open(OUT_IDMAP, "w") as f:
    json.dump(keys, f, indent=2)

print("Saved key mapping →", OUT_IDMAP)

print("\nDONE. FAISS index is ready.")
