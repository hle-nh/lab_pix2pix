import matplotlib.pyplot as plt
import numpy as np
from skimage.color import lab2rgb
import os

def save_comparison(L, fake_ab, real_ab, epoch, out_dir="visuals"):
    os.makedirs(out_dir, exist_ok=True)

    L = (L + 1) * 50
    fake_ab = fake_ab * 110
    real_ab = real_ab * 110

    n = L.size(0)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3*n))

    for i in range(n):
        gray = lab2rgb(
            np.stack([L[i,0], np.zeros_like(L[i,0]), np.zeros_like(L[i,0])], axis=-1)
        )
        fake = lab2rgb(
            np.stack([L[i,0], fake_ab[i,0], fake_ab[i,1]], axis=-1)
        )
        real = lab2rgb(
            np.stack([L[i,0], real_ab[i,0], real_ab[i,1]], axis=-1)
        )

        axes[i,0].imshow(gray)
        axes[i,1].imshow(fake)
        axes[i,2].imshow(real)

        for j in range(3):
            axes[i,j].axis("off")

    axes[0,0].set_title("Grayscale")
    axes[0,1].set_title("Generated")
    axes[0,2].set_title("Ground Truth")

    plt.tight_layout()
    plt.savefig(f"{out_dir}/epoch_{epoch:03d}.png")
    plt.close()
