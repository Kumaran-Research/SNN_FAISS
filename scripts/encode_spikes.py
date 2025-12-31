import os
import json
import numpy as np
from PIL import Image
from tqdm import tqdm

# ============================================================
# CONFIG — SNN spike generation
# ============================================================

T = 30                 # time steps (ms) — safe for CPU hardware
MAX_RATE = 40         # 100 Hz Poisson rate
IMG_SIZE = 32          # matches preprocess.py

# ============================================================
# PATHS
# ============================================================

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

PROC_DIR = os.path.join(ROOT, "data", "processed")

IMG_DIR = os.path.join(PROC_DIR, "images_resized")
CAP_EMB_PATH = os.path.join(PROC_DIR, "caption_embeddings.npy")
CAP_INDEX_PATH = os.path.join(PROC_DIR, "caption_index.json")

OUT_SPIKE_DIR = os.path.join(PROC_DIR, "spikes")
OUT_IMG_SPIKES = os.path.join(OUT_SPIKE_DIR, "images")
OUT_CAP_SPIKES = os.path.join(OUT_SPIKE_DIR, "captions")

os.makedirs(OUT_IMG_SPIKES, exist_ok=True)
os.makedirs(OUT_CAP_SPIKES, exist_ok=True)

# ============================================================
# POISSON SPIKE GENERATOR (with anti-collapse noise)
# ============================================================

def poisson_spike_train(rate_vec, T):
    rate_vec = rate_vec ** 2      # compress high intensities
    rate_vec = np.clip(rate_vec, 0, 1)

    # Poisson probability for each neuron
    prob = 0.5 * rate_vec         # lower baseline firing
    prob = np.clip(prob, 0, 1)

    # Base Poisson spikes
    spikes = np.random.rand(T, len(rate_vec)) < prob

    # Drop 3% of spikes randomly
    drop_mask = (np.random.rand(*spikes.shape) > 0.03)
    spikes = spikes * drop_mask


    # ----------------------------------------------------
    # FIX 7 — Add tiny stochastic noise to prevent collapse
    # ----------------------------------------------------
    noise = np.random.rand(T, len(rate_vec)) < 0.002  # 1% random spike firing
    spikes = np.logical_or(spikes, noise).astype(np.uint8)

    return spikes


# ============================================================
# MAIN EXECUTION
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # LOAD CAPTION EMBEDDINGS
    # --------------------------------------------------------
    print("\nLoading caption embeddings...")
    caption_embeddings = np.load(CAP_EMB_PATH)    # shape (N, EMB_DIM)

    with open(CAP_INDEX_PATH, "r") as f:
        cap_index_map = json.load(f)

    EMB_DIM = caption_embeddings.shape[1]
    print(f"Caption embedding dimension: {EMB_DIM}")

    # --------------------------------------------------------
    # IMAGE SPIKES
    # --------------------------------------------------------
    print("\nSTEP 1 — Encoding image spikes")
    for fname in tqdm(os.listdir(IMG_DIR)):

        img_path = os.path.join(IMG_DIR, fname)
        base = os.path.splitext(fname)[0]

        try:
            img = Image.open(img_path).convert("L")
        except:
            print("Skipping unreadable:", fname)
            continue

        # Normalize to [0,1]
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = arr.flatten()  # 32×32 = 1024

        spikes = poisson_spike_train(arr, T)
        np.save(os.path.join(OUT_IMG_SPIKES, base + ".jpg.npy"), spikes)

    # --------------------------------------------------------
    # CAPTION SPIKES
    # --------------------------------------------------------
    print("\nSTEP 2 — Encoding caption spikes")

    for img_name, idx in tqdm(cap_index_map.items()):

        emb = caption_embeddings[idx]

        # Normalize embedding to [0,1]
        max_abs = np.max(np.abs(emb)) + 1e-8
        norm_emb = (emb / max_abs + 1) / 2.0

        spikes = poisson_spike_train(norm_emb, T)

        # Ensure file name ends with .jpg.npy
        base = img_name
        if not base.lower().endswith(".jpg"):
            base = base + ".jpg"

        np.save(os.path.join(OUT_CAP_SPIKES, base + ".npy"), spikes)

    print("\n🎉 Spike encoding complete!")
    print(f"Image spikes saved → {OUT_IMG_SPIKES}")
    print(f"Caption spikes saved → {OUT_CAP_SPIKES}")
