import os
import json
import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC = os.path.join(ROOT, "data", "processed")

SPIKE_IMG_DIR = os.path.join(PROC, "spikes", "images")
SPIKE_CAP_DIR = os.path.join(PROC, "spikes", "captions")

CAP_EMB_PATH = os.path.join(PROC, "caption_embeddings.npy")
CAP_INDEX_PATH = os.path.join(PROC, "caption_index.json")

FINAL_EMB_PATH = os.path.join(PROC, "final_embeddings.npy")  # FIXED


def check_spike_file(path):
    arr = np.load(path)
    return arr.shape, arr.sum()


def main():

    print("\n=============================")
    print("  SNN DIAGNOSTIC REPORT")
    print("=============================\n")

    # ---------------------------------------------------------
    # 1. Check image spikes
    # ---------------------------------------------------------
    img_files = sorted(os.listdir(SPIKE_IMG_DIR))
    if len(img_files) == 0:
        print("❌ ERROR: No image spike files found.")
        return

    sample_img = os.path.join(SPIKE_IMG_DIR, img_files[0])
    shape, nonzero = check_spike_file(sample_img)
    print(f"Image spike sample: {img_files[0]}")
    print("Shape:", shape)
    print("Nonzero spikes:", nonzero)

    # ---------------------------------------------------------
    # 2. Check caption spikes
    # ---------------------------------------------------------
    cap_files = sorted(os.listdir(SPIKE_CAP_DIR))
    sample_cap = os.path.join(SPIKE_CAP_DIR, cap_files[0])
    shape, nonzero = check_spike_file(sample_cap)
    print("\nCaption spike sample:", cap_files[0])
    print("Shape:", shape)
    print("Nonzero spikes:", nonzero)

    # ---------------------------------------------------------
    # 3. Check caption embeddings
    # ---------------------------------------------------------
    print("\nLoading caption embeddings...")
    emb = np.load(CAP_EMB_PATH)
    print("Caption embedding shape:", emb.shape)
    print("Example values:", emb[0][:10])

    # ---------------------------------------------------------
    # 4. Check caption index
    # ---------------------------------------------------------
    print("\nLoading caption_index.json...")
    with open(CAP_INDEX_PATH, "r") as f:
        index = json.load(f)

    img_keys = list(index.keys())
    print("First 5 keys:", img_keys[:5])

    # ---------------------------------------------------------
    # 5. Check final embeddings (if exist)
    # ---------------------------------------------------------
    if os.path.exists(FINAL_EMB_PATH):
        print("\nLoading final_embeddings.npy...")
        final_emb = np.load(FINAL_EMB_PATH)

        print("Final embedding shape:", final_emb.shape)

        t = torch.tensor(final_emb).float()
        sim = F.cosine_similarity(t.unsqueeze(1), t.unsqueeze(0), dim=-1)

        print("Mean cosine similarity:", sim.mean().item())

        if sim.mean().item() > 0.8:
            print("⚠ WARNING: Embeddings highly similar — collapse possible.")
        else:
            print("✔ Embeddings show healthy variation.")
    else:
        print("\n⚠ No final_embeddings.npy yet — run training first.")

    print("\n=============================")
    print("     DIAGNOSTIC COMPLETE")
    print("=============================\n")


if __name__ == "__main__":
    main()
