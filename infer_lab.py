import os
import random
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from skimage import color

from models_lab import GeneratorUNet
from config import DEVICE

# ==========================
# CONFIG
# ==========================
VAL_DIR = "C:/lab_pix2pix/val2017"
CHECKPOINT = "checkpoints/G_baseline.pt" #default
NUM_IMAGES = 15
OUT_DIR = "infer_outputs"

IMG_SIZE = 256

os.makedirs(OUT_DIR, exist_ok=True)

# ==========================
# TRANSFORM (GIỐNG TRAIN)
# ==========================
tf = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor()
])

# ==========================
# UTILS
# ==========================
def is_grayscale(img_rgb: torch.Tensor) -> bool:
    """Check ảnh RGB thật hay grayscale giả"""
    return torch.allclose(img_rgb[0], img_rgb[1]) and \
           torch.allclose(img_rgb[1], img_rgb[2])

# ==========================
# INFER ONE IMAGE
# ==========================
def infer_one(image_path, model, out_path):
    # ---- Load GT RGB ----
    img_rgb = Image.open(image_path).convert("RGB")
    img_rgb = tf(img_rgb)

    # ---- Skip grayscale GT ----
    if is_grayscale(img_rgb):
        return False

    # ---- Create grayscale input ----
    gray = T.functional.rgb_to_grayscale(img_rgb, num_output_channels=1)

    # ---- LAB pipeline ----
    rgb_fake = gray.repeat(3, 1, 1)
    lab = color.rgb2lab(rgb_fake.permute(1, 2, 0).numpy())

    L = torch.from_numpy(lab[:, :, 0]).unsqueeze(0).unsqueeze(0)
    L = L / 50.0 - 1.0
    L = L.to(DEVICE)

    with torch.no_grad():
        fake_ab = model(L)

    # ---- LAB → RGB ----
    lab_out = torch.cat([L.cpu(), fake_ab.cpu()], dim=1)
    lab_out[:, 0] = (lab_out[:, 0] + 1) * 50
    lab_out[:, 1:] = lab_out[:, 1:] * 110

    rgb_out = color.lab2rgb(
        lab_out.permute(0, 2, 3, 1).numpy()
    )
    rgb_out = torch.from_numpy(rgb_out).permute(0, 3, 1, 2)
    rgb_out = torch.clamp(rgb_out, 0, 1)

    # ---- Prepare comparison ----
    gray_vis = gray.repeat(3, 1, 1).unsqueeze(0)
    gt_vis   = img_rgb.unsqueeze(0)

    comparison = torch.cat(
        [gray_vis, rgb_out, gt_vis],
        dim=0
    )

    save_image(comparison, out_path, nrow=3)
    return True

# ==========================
# MAIN
# ==========================
def main():
    # ---- Load model ----
    model = GeneratorUNet().to(DEVICE)
    model.load_state_dict(
        torch.load(CHECKPOINT, map_location=DEVICE)
    )
    model.eval()

    # ---- Collect images ----
    files = [
        f for f in os.listdir(VAL_DIR)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]
    random.shuffle(files)

    saved = 0
    for fname in files:
        if saved >= NUM_IMAGES:
            break

        in_path = os.path.join(VAL_DIR, fname)
        out_path = os.path.join(OUT_DIR, fname)

        ok = infer_one(in_path, model, out_path)
        if ok:
            saved += 1
            print(f"Saved {saved}/{NUM_IMAGES}: {fname}")

    print(f"\nDone. Saved {saved} images to → {OUT_DIR}")

if __name__ == "__main__":
    main()
#
