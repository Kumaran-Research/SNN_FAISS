import os
import json
from PIL import Image
import numpy as np
from tqdm import tqdm
import re

# ============================================================
# CONFIG — minimal, SNN-friendly
# ============================================================

IMG_SIZE = 32
EMB_DIM = 64      # tiny, stable embedding dimension

# ============================================================
# PATH SETUP
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
RAW_IMG_DIR = os.path.join(ROOT, "data", "raw", "images")
RAW_CAP_PATH = os.path.join(ROOT, "data", "raw", "captions.txt")

PROC_DIR = os.path.join(ROOT, "data", "processed")
os.makedirs(PROC_DIR, exist_ok=True)

OUT_IMG_DIR = os.path.join(PROC_DIR, "images_resized")
os.makedirs(OUT_IMG_DIR, exist_ok=True)

OUT_CAP_EMB = os.path.join(PROC_DIR, "caption_embeddings.npy")
OUT_CAP_INDEX = os.path.join(PROC_DIR, "caption_index.json")

# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(t):
    t = t.lower()
    t = re.sub(r"[^a-z0-9.,!? ]+", "", t)
    return t.strip()

# ============================================================
# STABLE RANDOM WORD EMBEDDING TABLE
# (same word → same vector across runs)
# ============================================================

random_table = {}

def word_embedding(word):
    if word not in random_table:
        random_table[word] = np.random.uniform(-1, 1, EMB_DIM)
    return random_table[word]

# ============================================================
# SENTENCE → TINY SEMANTIC VECTOR
# ============================================================

def embed_caption(sentence):
    words = clean_text(sentence).split()
    if len(words) == 0:
        return np.zeros(EMB_DIM, dtype=np.float32)

    vecs = [word_embedding(w) for w in words]
    return np.mean(vecs, axis=0).astype(np.float32)

# ============================================================
# MAIN PIPELINE
# ============================================================

if __name__ == "__main__":

    print("\nSTEP 1 — Resize images (32×32 grayscale)")
    for fname in tqdm(os.listdir(RAW_IMG_DIR)):
        inp = os.path.join(RAW_IMG_DIR, fname)
        outp = os.path.join(OUT_IMG_DIR, fname)

        try:
            img = Image.open(inp).convert("L")
            img = img.resize((IMG_SIZE, IMG_SIZE))
            img.save(outp)
        except:
            print("Skipping:", fname)

    print("\nSTEP 2 — Load captions and embed")
    caption_vectors = []
    index_map = {}

    with open(RAW_CAP_PATH, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line or "\t" not in line:
                continue

            img, cap = line.split("\t", 1)
            img = img.strip()
if not img.endswith(".jpg"):
    img = img + ".jpg"

            cap = clean_text(cap)

            # Compute embedding
            emb = embed_caption(cap)
            caption_vectors.append(emb)
            index_map[img] = idx

    caption_vectors = np.vstack(caption_vectors)
    np.save(OUT_CAP_EMB, caption_vectors)

    with open(OUT_CAP_INDEX, "w") as f:
        json.dump(index_map, f, indent=2)

    print(f"\n✔ Saved resized images → {OUT_IMG_DIR}")
    print(f"✔ Saved caption embeddings → {OUT_CAP_EMB}")
    print(f"✔ Saved caption index → {OUT_CAP_INDEX}")
    print("\n🎉 Preprocessing complete!")
