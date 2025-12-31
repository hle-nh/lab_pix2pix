# train_subset_cached_fast.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb

from models_lab import GeneratorUNet, PatchDiscriminator

# ==========================
# CONFIG
# ==========================
CACHE_DIR = "lab_cache"          # <-- output of cache_lab.py
IMG_SIZE = 256

EPOCHS = 30
BATCH_SIZE = 16
LR = 2e-4
LAMBDA_L1 = 100

NUM_WORKERS = 0                 # safe on Windows
PIN_MEMORY = True
VIS_EVERY = 2

CHECKPOINT_DIR = "checkpoints_subset"
VIS_DIR = "visuals"
LOG_DIR = "logs"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# DATASET (CACHED LAB)
# ==========================
class CachedLABDataset(Dataset):
    def __init__(self, cache_dir):
        self.files = [
            os.path.join(cache_dir, f)
            for f in os.listdir(cache_dir)
            if f.endswith(".pt")
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        sample = torch.load(self.files[idx], map_location="cpu")
        return sample["L"], sample["ab"]

# ==========================
# FIXED SAMPLES FOR VISUALS
# ==========================
def get_fixed_samples(dataset, n=5):
    idxs = torch.linspace(0, len(dataset) - 1, n).long()
    samples = [dataset[i] for i in idxs]
    Ls, abs_ = zip(*samples)
    return torch.stack(Ls), torch.stack(abs_)

# ==========================
# VISUALIZATION
# ==========================
def save_visuals(L, fake_ab, real_ab, epoch):
    os.makedirs(VIS_DIR, exist_ok=True)

    # denormalize
    L = (L + 1) * 50
    fake_ab = fake_ab * 110
    real_ab = real_ab * 110

    n = L.size(0)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))

    for i in range(n):
        gray = lab2rgb(np.stack([L[i, 0], np.zeros_like(L[i, 0]), np.zeros_like(L[i, 0])], -1))
        fake = lab2rgb(np.stack([L[i, 0], fake_ab[i, 0], fake_ab[i, 1]], -1))
        real = lab2rgb(np.stack([L[i, 0], real_ab[i, 0], real_ab[i, 1]], -1))

        axes[i, 0].imshow(gray)
        axes[i, 1].imshow(fake)
        axes[i, 2].imshow(real)

        for j in range(3):
            axes[i, j].axis("off")

    axes[0, 0].set_title("Grayscale")
    axes[0, 1].set_title("Generated")
    axes[0, 2].set_title("Ground Truth")

    plt.tight_layout()
    plt.savefig(f"{VIS_DIR}/epoch_{epoch:03d}.png")
    plt.close()

# ==========================
# TRAINING
# ==========================
def train():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    dataset = CachedLABDataset(CACHE_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    fixed_L, fixed_ab = get_fixed_samples(dataset, n=5)
    fixed_L = fixed_L.to(DEVICE)
    fixed_ab = fixed_ab.to(DEVICE)

    G = GeneratorUNet().to(DEVICE)
    D = PatchDiscriminator().to(DEVICE)

    opt_G = torch.optim.Adam(G.parameters(), LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), LR, betas=(0.5, 0.999))

    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    scaler = GradScaler("cuda")

    losses_G, losses_D = [], []

    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Epoch {epoch}/{EPOCHS}")
        g_epoch, d_epoch = 0.0, 0.0

        G.train()
        D.train()

        for L, ab in pbar:
            L = L.to(DEVICE, non_blocking=True)
            ab = ab.to(DEVICE, non_blocking=True)

            # ---- Train Discriminator ----
            opt_D.zero_grad()
            with autocast("cuda"):
                fake_ab = G(L).detach()
                loss_D = 0.5 * (
                    criterion_gan(D(L, ab), torch.ones_like(D(L, ab))) +
                    criterion_gan(D(L, fake_ab), torch.zeros_like(D(L, fake_ab)))
                )
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)

            # ---- Train Generator ----
            opt_G.zero_grad()
            with autocast("cuda"):
                fake_ab = G(L)
                loss_G = (
                    criterion_gan(D(L, fake_ab), torch.ones_like(D(L, fake_ab))) +
                    criterion_l1(fake_ab, ab) * LAMBDA_L1
                )
            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()

            g_epoch += loss_G.item()
            d_epoch += loss_D.item()

            pbar.set_postfix(G=loss_G.item(), D=loss_D.item())

        losses_G.append(g_epoch / len(loader))
        losses_D.append(d_epoch / len(loader))

        # ---- Visuals ----
        if epoch % VIS_EVERY == 0:
            G.eval()
            with torch.no_grad():
                fake_fixed = G(fixed_L)
                save_visuals(fixed_L.cpu(), fake_fixed.cpu(), fixed_ab.cpu(), epoch)
            G.train()

        torch.save(
            G.state_dict(),
            f"{CHECKPOINT_DIR}/G_epoch_{epoch:03d}.pt"
        )

    # ==========================
    # LOSS PLOT
    # ==========================
    plt.figure()
    plt.plot(losses_G, label="Generator")
    plt.plot(losses_D, label="Discriminator")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Pix2Pix LAB Training Loss")
    plt.savefig(f"{LOG_DIR}/losses.png")
    plt.close()

# ==========================
# ENTRY POINT (WINDOWS SAFE)
# ==========================
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train()
