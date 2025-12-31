import os
import cv2
import json
import torch
import numpy as np
from tqdm import tqdm

from torchmetrics.image import (
    StructuralSimilarityIndexMeasure,
    MultiScaleStructuralSimilarityIndexMeasure
)
from torchmetrics.image.fid import FrechetInceptionDistance

# ==========================
# CONFIG
# ==========================
INPUT_DIR = "test2017"
MAX_IMAGES = 10000

OUTPUT_DIRS = {
    "Subset": "output_subset",
    "Finetune": "output_finetune",
    "Finetune+VGG": "output_finetune_vgg",
}

IMG_SIZE = 256
VALID_EXT = (".jpg", ".jpeg", ".png")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# METRIC HELPERS
# ==========================
def calculate_psnr(x, y):
    """
    x, y: torch tensor [1,3,H,W] in [0,1]
    """
    mse = torch.mean((x - y) ** 2)
    if mse < 1e-10:
        return 100.0
    return 20 * torch.log10(1.0 / torch.sqrt(mse)).item()


def calculate_colorfulness(img):
    """
    img: torch tensor [3,H,W] in [0,1]
    return: colorfulness score (Hasler–Süsstrunk, standard scale)
    """
    img = img.permute(1, 2, 0).cpu().numpy() * 255.0  #  back to [0,255]

    R, G, B = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    rg = R - G
    yb = 0.5 * (R + G) - B

    std_rg = rg.std()
    std_yb = yb.std()
    mean_rg = rg.mean()
    mean_yb = yb.mean()

    return np.sqrt(std_rg**2 + std_yb**2) + \
           0.3 * np.sqrt(mean_rg**2 + mean_yb**2)

# ==========================
# MAIN
# ==========================
if __name__ == "__main__":

    filenames = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(VALID_EXT)
    ])[:MAX_IMAGES]

    print(f"Evaluating {len(filenames)} images")

    all_results = {}

    # ==========================
    # EVALUATE EACH OUTPUT FOLDER
    # ==========================
    for name, out_dir in OUTPUT_DIRS.items():
        print(f"\n===== Evaluating {name} =====")

        ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        ms_ssim = MultiScaleStructuralSimilarityIndexMeasure(data_range=1.0).to(DEVICE)
        fid = FrechetInceptionDistance(feature=2048, normalize=True).to(DEVICE)

        psnr_list = []
        cf_list = []

        pbar = tqdm(
            filenames,
            desc=f"{name} | Evaluating",
            total=len(filenames),
            ncols=110
        )

        for idx, fname in enumerate(pbar, start=1):
            # ---- Load REAL ----
            real = cv2.imread(os.path.join(INPUT_DIR, fname))
            real = cv2.cvtColor(real, cv2.COLOR_BGR2RGB)
            real = cv2.resize(real, (IMG_SIZE, IMG_SIZE))
            real = torch.from_numpy(real / 255.0).permute(2, 0, 1).unsqueeze(0).float()

            # ---- Load FAKE ----
            fake = cv2.imread(os.path.join(out_dir, fname))
            fake = cv2.cvtColor(fake, cv2.COLOR_BGR2RGB)
            fake = cv2.resize(fake, (IMG_SIZE, IMG_SIZE))
            fake = torch.from_numpy(fake / 255.0).permute(2, 0, 1).unsqueeze(0).float()

            real = real.to(DEVICE)
            fake = fake.to(DEVICE)

            # ---- Metrics ----
            psnr_val = calculate_psnr(fake, real)
            psnr_list.append(psnr_val)

            ssim.update(fake, real)
            ms_ssim.update(fake, real)
            fid.update(real, real=True)
            fid.update(fake, real=False)

            cf_val = calculate_colorfulness(fake[0])
            cf_list.append(cf_val)

            # ---- Live status every 100 imgs ----
            if idx % 100 == 0:
                pbar.set_postfix(
                    PSNR=f"{np.mean(psnr_list):.2f}",
                    CF=f"{np.mean(cf_list):.1f}"
                )

            # ---- Free GPU memory ----
            del real, fake
            torch.cuda.empty_cache()

        results = {
            "psnr": float(np.mean(psnr_list)),
            "ssim": ssim.compute().item(),
            "ms_ssim": ms_ssim.compute().item(),
            "colorfulness": float(np.mean(cf_list)),
            "fid": fid.compute().item()
        }

        all_results[name] = results

        # ---- Per-model print ----
        print("PSNR:", results["psnr"])
        print("SSIM:", results["ssim"])
        print("MS-SSIM:", results["ms_ssim"])
        print("Colorfulness:", results["colorfulness"])
        print("FID:", results["fid"])

    # ==========================
    # TABULAR OUTPUT
    # ==========================
    print("\n================ TABULAR RESULTS ================\n")
    print(f"{'Model':15} {'PSNR':>8} {'SSIM':>8} {'MS-SSIM':>10} {'Color':>10} {'FID':>8}")
    print("-" * 72)

    for name, m in all_results.items():
        print(
            f"{name:15} "
            f"{m['psnr']:>8.2f} "
            f"{m['ssim']:>8.3f} "
            f"{m['ms_ssim']:>10.3f} "
            f"{m['colorfulness']:>10.1f} "
            f"{m['fid']:>8.2f}"
        )

    print("\n=================================================\n")

    # ==========================
    # SAVE JSON
    # ==========================
    with open("evaluation_metrics.json", "w") as f:
        json.dump(all_results, f, indent=4)
