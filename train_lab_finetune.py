# train_finetune_full.py
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
from skimage.color import rgb2lab

from models_lab import GeneratorUNet, PatchDiscriminator

# ==========================
# CONFIG
# ==========================
TRAIN_DIR = "C:/lab_pix2pix/train2017"
PRETRAINED_G = "checkpoints_subset/G_epoch_030.pt"

EPOCHS = 8
BATCH_SIZE = 4
LR = 5e-5
LAMBDA_L1 = 50

NUM_WORKERS = 0        # Windows-safe
PIN_MEMORY = True

CHECKPOINT_DIR = "checkpoints_finetune"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================
# DATASET (RGB → LAB ON THE FLY)
# ==========================
class CocoLABDataset(Dataset):
    def __init__(self, root):
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        self.tf = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.tf(img)

        lab = rgb2lab(img.permute(1, 2, 0).numpy())
        L  = torch.from_numpy(lab[:, :, 0]).unsqueeze(0) / 50.0 - 1.0
        ab = torch.from_numpy(lab[:, :, 1:]).permute(2, 0, 1) / 110.0

        return L.float(), ab.float()

# ==========================
# TRAIN
# ==========================
def train():
    dataset = CocoLABDataset(TRAIN_DIR)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY
    )

    # Models
    G = GeneratorUNet().to(DEVICE)
    D = PatchDiscriminator().to(DEVICE)

    # Load pretrained generator
    G.load_state_dict(torch.load(PRETRAINED_G, map_location=DEVICE))
    print(f"Loaded pretrained generator: {PRETRAINED_G}")

    opt_G = torch.optim.Adam(G.parameters(), LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), LR, betas=(0.5, 0.999))

    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()

    scaler = GradScaler("cuda")

    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(loader, desc=f"Finetune Epoch {epoch}/{EPOCHS}")

        for L, ab in pbar:
            L = L.to(DEVICE, non_blocking=True)
            ab = ab.to(DEVICE, non_blocking=True)

            # ---- Train D ----
            opt_D.zero_grad()
            with autocast("cuda"):
                fake_ab = G(L).detach()
                loss_D = 0.5 * (
                    criterion_gan(D(L, ab), torch.ones_like(D(L, ab))) +
                    criterion_gan(D(L, fake_ab), torch.zeros_like(D(L, fake_ab)))
                )
            scaler.scale(loss_D).backward()
            scaler.step(opt_D)

            # ---- Train G ----
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

            pbar.set_postfix(G=loss_G.item(), D=loss_D.item())

        torch.save(
            G.state_dict(),
            f"{CHECKPOINT_DIR}/G_finetune_epoch_{epoch:02d}.pt"
        )

    print("Fine-tuning finished.")

# ==========================
# ENTRY POINT
# ==========================
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train()
