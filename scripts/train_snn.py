# scripts/train_snn.py

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
import yaml

from src.models.snn_model import SNNEncoder

# ============================================================
# CONFIG
# ============================================================
# Load configuration
with open("configs/config.yaml", "r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Extract training parameters
T = config["data"]["time_steps"]
BATCH_SIZE = config["training"]["batch_size"]
EPOCHS = config["training"]["epochs"]
LR = config["training"]["learning_rate"]
EMBED_DIM = config["training"]["embedding_dim"]
TEMP = config["training"]["temperature"]

# ============================================================
# PATHS
# ============================================================
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PROC = os.path.join(ROOT, "data", "processed")

IMG_SPIKE_DIR = os.path.join(PROC, "spikes", "images")
CAP_SPIKE_DIR = os.path.join(PROC, "spikes", "captions")

CAP_INDEX_PATH = os.path.join(PROC, "caption_index.json")

FINAL_EMB_PATH = os.path.join(PROC, "final_embeddings.npy")
FINAL_KEY_PATH = os.path.join(PROC, "final_keys.json")

# ============================================================
# DATASET
# ============================================================

class SpikeEpisodeDataset(Dataset):
    def __init__(self, img_dir, cap_dir, index_json):
        self.img_dir = img_dir
        self.cap_dir = cap_dir

        with open(index_json, "r") as f:
            idx_map = json.load(f)

        # build pairs
        self.pairs = []

        for fname, idx in idx_map.items():
            spike_img = os.path.join(img_dir, fname + ".npy")
            spike_cap = os.path.join(cap_dir, fname + ".npy")

            if os.path.exists(spike_img) and os.path.exists(spike_cap):
                self.pairs.append((spike_img, spike_cap))

        if len(self.pairs) == 0:
            raise RuntimeError("No usable spike pairs found. Check naming consistency.")

        print("Dataset size:", len(self.pairs))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img_path, cap_path = self.pairs[idx]
        img = np.load(img_path)  # (T, D)
        cap = np.load(cap_path)

        return torch.from_numpy(img), torch.from_numpy(cap)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train():
    dataset = SpikeEpisodeDataset(IMG_SPIKE_DIR, CAP_SPIKE_DIR, CAP_INDEX_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    # Find dimensions from a sample
    sample_img, sample_cap = dataset[0]
    img_in_dim = sample_img.shape[1]
    cap_in_dim = sample_cap.shape[1]

    img_encoder = SNNEncoder(
        in_dim=img_in_dim,
        hidden_dims=(256, 128),
        embedding_dim=EMBED_DIM,
        dropout=0.1,
        read_k_fraction=0.2
    ).to(DEVICE)

    cap_encoder = SNNEncoder(
        in_dim=cap_in_dim,
        hidden_dims=(256, 128),
        embedding_dim=EMBED_DIM,
        dropout=0.1,
        read_k_fraction=0.2
    ).to(DEVICE)

    opt = torch.optim.AdamW(
        list(img_encoder.parameters()) +
        list(cap_encoder.parameters()),
        lr=LR,
        weight_decay=1e-4,
        betas=(0.9, 0.95)
    )

    for epoch in range(EPOCHS):
        img_encoder.train()
        cap_encoder.train()

        epoch_loss = 0.0

        for img_spikes, cap_spikes in tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):

            img_spikes = img_spikes.to(DEVICE)
            cap_spikes = cap_spikes.to(DEVICE)

            opt.zero_grad()

            # Forward
            z_img = img_encoder(img_spikes)
            z_cap = cap_encoder(cap_spikes)

            z_img = F.normalize(z_img, p=2, dim=1)
            z_cap = F.normalize(z_cap, p=2, dim=1)

            # Contrastive alignment loss
            logits = (z_img @ z_cap.t()) / TEMP
            labels = torch.arange(len(img_spikes), device=DEVICE)

            loss = F.cross_entropy(logits, labels)

            # Stability check
            if torch.isnan(loss) or torch.isinf(loss):
                print("WARNING: NaN/Inf detected. Skipping batch.")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(img_encoder.parameters(), 1.0)
            torch.nn.utils.clip_grad_norm_(cap_encoder.parameters(), 1.0)
            opt.step()

            epoch_loss += loss.item()

        print(f"Epoch {epoch+1}/{EPOCHS} — Loss: {epoch_loss/len(loader):.4f}")

    print("Training complete.")

        # ============================================================
    # ENCODE FULL DATASET (NO GRAD, SAFE)
    # ============================================================

    print("Encoding full dataset embeddings...")

    img_encoder.eval()
    cap_encoder.eval()

    all_embeddings = []
    all_keys = []

    with torch.no_grad():                      # <---- FIX #1
        for i in tqdm(range(len(dataset))):
            img_spikes, cap_spikes = dataset[i]

            img_spikes = img_spikes.unsqueeze(0).to(DEVICE)
            cap_spikes = cap_spikes.unsqueeze(0).to(DEVICE)

            z_img = img_encoder(img_spikes)
            z_cap = cap_encoder(cap_spikes)

            emb = F.normalize((z_img + z_cap) / 2, p=2, dim=1)

            # FIX #2 — detach before numpy
            all_embeddings.append(
                emb.squeeze(0).detach().cpu().numpy()
            )

            fname = os.path.basename(dataset.pairs[i][0]).replace(".npy", "")
            all_keys.append(fname)

    all_embeddings = np.stack(all_embeddings, axis=0)
    np.save(FINAL_EMB_PATH, all_embeddings)

    with open(FINAL_KEY_PATH, "w") as f:
        json.dump(all_keys, f)

    print(f"Saved final embeddings → {FINAL_EMB_PATH}")
    print(f"Saved key mapping → {FINAL_KEY_PATH}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    train()
