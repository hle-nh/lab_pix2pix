import os
import json
import shutil
import cv2
import numpy as np
from tqdm import tqdm

# ======================
# PATHS (EDIT THESE)
# ======================
COCO_ROOT = "C:/lab_pix2pix"
ANN_FILE = os.path.join(COCO_ROOT, "annotations/instances_train2017.json")
IMG_DIR  = os.path.join(COCO_ROOT, "train2017")
OUT_DIR  = "coco_subset_colorization"

os.makedirs(OUT_DIR, exist_ok=True)

# ======================
# FILTER PARAMETERS
# ======================
ALLOWED_CATEGORIES = {
    "person",
    "dog", "cat", "horse", "cow", "sheep",
    "bird", "elephant", "giraffe", "zebra",
    "car", "bus", "truck", "bicycle", "motorcycle"
}

MIN_OBJ_AREA_RATIO = 0.15     # ≥15% of image
MIN_MEAN_SAT       = 0.15
MIN_L_MEAN         = 30
MAX_L_MEAN         = 85
MAX_OBJECTS        = 5        # scene simplicity

MAX_IMAGES         = 30000    # safety cap

# ======================
# LOAD COCO ANNOTATIONS
# ======================
with open(ANN_FILE, "r") as f:
    coco = json.load(f)

# category id → name
cat_id_to_name = {
    cat["id"]: cat["name"] for cat in coco["categories"]
}

# image_id → image info
image_info = {
    img["id"]: img for img in coco["images"]
}

# image_id → annotations
image_to_anns = {}
for ann in coco["annotations"]:
    image_to_anns.setdefault(ann["image_id"], []).append(ann)

# ======================
# FILTER LOOP
# ======================
kept = 0

for image_id, anns in tqdm(image_to_anns.items(), desc="Filtering COCO"):

    if kept >= MAX_IMAGES:
        break

    img_meta = image_info[image_id]
    file_name = img_meta["file_name"]
    img_path = os.path.join(IMG_DIR, file_name)

    if not os.path.exists(img_path):
        continue

    # ------------------
    # Scene complexity
    # ------------------
    if len(anns) > MAX_OBJECTS:
        continue

    # ------------------
    # Category + size
    # ------------------
    img_area = img_meta["width"] * img_meta["height"]
    keep_image = False

    for ann in anns:
        cat_name = cat_id_to_name[ann["category_id"]]
        if cat_name not in ALLOWED_CATEGORIES:
            continue

        bbox_area = ann["bbox"][2] * ann["bbox"][3]
        if bbox_area / img_area >= MIN_OBJ_AREA_RATIO:
            keep_image = True
            break

    if not keep_image:
        continue

    # ------------------
    # Load image
    # ------------------
    img = cv2.imread(img_path)
    if img is None:
        continue

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # ------------------
    # Saturation filter
    # ------------------
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mean_sat = hsv[..., 1].mean() / 255.0
    if mean_sat < MIN_MEAN_SAT:
        continue

    # ------------------
    # Luminance filter
    # ------------------
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    mean_L = lab[..., 0].mean() * 100.0 / 255.0
    if not (MIN_L_MEAN <= mean_L <= MAX_L_MEAN):
        continue

    # ------------------
    # KEEP IMAGE
    # ------------------
    shutil.copy(img_path, os.path.join(OUT_DIR, file_name))
    kept += 1

print(f"\n Kept {kept} images in subset → {OUT_DIR}")
