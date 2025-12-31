# cache_lab.py
import os
import torch
from PIL import Image
from torchvision import transforms
from skimage.color import rgb2lab
from tqdm import tqdm

# --------------------
# CONFIG
# --------------------
IMG_DIR = "coco_subset_colorization"
CACHE_DIR = "lab_cache"
IMG_SIZE = 256

os.makedirs(CACHE_DIR, exist_ok=True)

tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor()
])

# --------------------
# CACHE LOOP
# --------------------
files = [
    f for f in os.listdir(IMG_DIR)
    if f.lower().endswith((".jpg", ".png", ".jpeg"))
]

for fname in tqdm(files, desc="Caching LAB"):
    img = Image.open(os.path.join(IMG_DIR, fname)).convert("RGB")
    img = tf(img)

    lab = rgb2lab(img.permute(1, 2, 0).numpy())
    L  = torch.from_numpy(lab[:, :, 0]).unsqueeze(0) / 50.0 - 1.0
    ab = torch.from_numpy(lab[:, :, 1:]).permute(2, 0, 1) / 110.0

    torch.save(
        {"L": L.float(), "ab": ab.float()},
        os.path.join(CACHE_DIR, fname.replace(".jpg", ".pt").replace(".png", ".pt"))
    )

print("✅ LAB cache created")
