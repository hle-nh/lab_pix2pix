import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torch.amp import autocast, GradScaler
from tqdm import tqdm
from PIL import Image
from skimage.color import rgb2lab, lab2rgb

from models_lab import GeneratorUNet, PatchDiscriminator
from config import DEVICE

# ==========================
# CONFIG
# ==========================
TRAIN_DIR = "C:/lab_pix2pix/train2017"

PRETRAINED_G = "checkpoints_finetune/G_finetune_epoch_04.pt"

CHECKPOINT_DIR = "checkpoints_finetune"

# Start finetune + vgg after epoch 4 of normal finetune
START_EPOCH = 4

# VGG finetune
EPOCHS = 1
BATCH_SIZE = 4
LR = 2e-5

LAMBDA_L1 = 30
LAMBDA_VGG = 1.0

NUM_WORKERS = 0
PIN_MEMORY = True

IMG_SIZE = 256

os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# ==========================
# DATASET
# ==========================
class CocoLABDataset(Dataset):
    def __init__(self, root):
        self.files = [
            os.path.join(root, f)
            for f in os.listdir(root)
            if f.lower().endswith((".jpg", ".png", ".jpeg"))
        ]
        self.tf = transforms.Compose([
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img = Image.open(self.files[idx]).convert("RGB")
        img = self.tf(img)

        lab = rgb2lab(img.permute(1, 2, 0).numpy())

        L = torch.from_numpy(lab[:, :, 0]).unsqueeze(0) / 50.0 - 1.0
        ab = torch.from_numpy(lab[:, :, 1:]).permute(2, 0, 1) / 110.0

        return L.float(), ab.float()

# ==========================
# VGG PERCEPTUAL LOSS
# ==========================
class VGGPerceptualLoss(nn.Module):
    def __init__(self):
        super().__init__()
        vgg = models.vgg16(
            weights=models.VGG16_Weights.IMAGENET1K_V1
        ).features

        self.slice = nn.Sequential(*list(vgg.children())[:16]).eval()
        for p in self.slice.parameters():
            p.requires_grad = False

    def forward(self, x, y):
        # x, y ∈ [0,1], RGB
        return nn.functional.l1_loss(self.slice(x), self.slice(y))

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
    G.load_state_dict(
        torch.load(PRETRAINED_G, map_location=DEVICE)
    )
    print(f"Loaded generator: {PRETRAINED_G}")

    opt_G = torch.optim.Adam(G.parameters(), LR, betas=(0.5, 0.999))
    opt_D = torch.optim.Adam(D.parameters(), LR, betas=(0.5, 0.999))

    criterion_gan = nn.MSELoss()
    criterion_l1 = nn.L1Loss()
    criterion_vgg = VGGPerceptualLoss().to(DEVICE)

    scaler = GradScaler("cuda")

    # ==========================
    # TRAIN LOOP
    # ==========================
    for epoch in range(1, EPOCHS + 1):
        pbar = tqdm(
            loader,
            desc=f"VGG Finetune Epoch {epoch}/{EPOCHS}"
        )

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

                loss_gan = criterion_gan(
                    D(L, fake_ab),
                    torch.ones_like(D(L, fake_ab))
                )

                loss_l1 = criterion_l1(fake_ab, ab) * LAMBDA_L1

                # LAB → RGB (cho perceptual)
                lab_fake = torch.cat([L, fake_ab], dim=1)
                lab_fake[:, 0] = (lab_fake[:, 0] + 1) * 50
                lab_fake[:, 1:] = lab_fake[:, 1:] * 110

                rgb_fake = torch.from_numpy(
                    lab2rgb(
                        lab_fake.permute(0, 2, 3, 1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                ).permute(0, 3, 1, 2).to(DEVICE)

                lab_gt = torch.cat([L, ab], dim=1)
                lab_gt[:, 0] = (lab_gt[:, 0] + 1) * 50
                lab_gt[:, 1:] = lab_gt[:, 1:] * 110

                rgb_gt = torch.from_numpy(
                    lab2rgb(
                        lab_gt.permute(0, 2, 3, 1)
                        .detach()
                        .cpu()
                        .numpy()
                    )
                ).permute(0, 3, 1, 2).to(DEVICE)

                loss_vgg = criterion_vgg(rgb_fake, rgb_gt) * LAMBDA_VGG

                loss_G = loss_gan + loss_l1 + loss_vgg

            scaler.scale(loss_G).backward()
            scaler.step(opt_G)
            scaler.update()

            pbar.set_postfix(
                G=loss_G.item(),
                L1=loss_l1.item(),
                VGG=loss_vgg.item()
            )

        # ---- Save checkpoint (PHÂN BIỆT RÕ) ----
        torch.save(
            G.state_dict(),
            f"{CHECKPOINT_DIR}/G_ft_fromE{START_EPOCH}_vgg_epoch_{epoch:02d}.pt"
        )

    print("VGG fine-tuning finished.")

# ==========================
# ENTRY POINT
# ==========================
if __name__ == "__main__":
    torch.multiprocessing.freeze_support()
    train()
