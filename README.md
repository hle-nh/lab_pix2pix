# Image Colorization using Pix2Pix (LAB Color Space)

This project implements an **automatic image colorization system** based on a **conditional GAN (Pix2Pix)** operating in **CIELAB color space**.
The model learns to predict chrominance (ab) channels from grayscale luminance (L) and is trained progressively on the **COCO 2017 dataset**.

The pipeline includes:

* Subset training for stable initialization
* Full-dataset fine-tuning
* Perceptual fine-tuning using VGG loss
* Inference, visualization, and quantitative evaluation

---

## Project Overview

The system takes a **grayscale image** as input and produces a **plausible colorized RGB image** as output.

Key characteristics:

* Generator: U-Net–based Pix2Pix generator
* Discriminator: PatchGAN discriminator
* Color space: LAB (L for structure, ab for color)
* Losses: GAN loss, L1 loss, optional VGG perceptual loss
* Dataset: COCO 2017

---

## Project Structure

```
lab_pix2pix/
├── models_lab.py                 # Generator (U-Net) & PatchGAN Discriminator
├── config.py                     # Global configuration (DEVICE, paths)
│
├── dataset_lab_subset.py         # Dataset loader for subset training
├── dataset_lab_cache.py          # Dataset loader using cached LAB tensors
├── cache_lab.py                  # Script to precompute LAB cache
├── filter.py                     # Filter images (e.g. remove grayscale)
│
├── train_lab.py                  # Train Pix2Pix on subset
├── train_lab_finetune.py         # Fine-tune on full dataset
├── train_vgg.py                  # Fine-tune with VGG perceptual loss
│
├── infer_lab.py                  # Inference (grayscale → color)
│
├── utils/
│   ├── visualization.py          # Qualitative comparison utilities
│   └── fixed_samples.py          # Fixed images for consistent visualization
│
├── calculate_metrics.py          # Quantitative evaluation script
│
├── requirements.txt
└── README.md
```

---

## Requirements

* Python 3.8+
* CUDA-capable GPU (recommended)
* PyTorch + torchvision
* torchmetrics, torch-fidelity (for FID)
* OpenCV, scikit-image, numpy

---

## Installation

### 1. Create virtual environment (recommended)

```bash
python -m venv venv
```

Activate:

* **Windows**: `venv\Scripts\activate`
* **Linux / macOS**: `source venv/bin/activate`

---

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

If using CUDA, make sure your PyTorch version matches your CUDA version.

---

## Dataset Setup

Expected dataset structure (COCO-style):

```
C:/lab_pix2pix/
├── train2017/
├── val2017/        # optional
├── test2017/
```

All images are RGB.
Grayscale inputs are generated dynamically during training and inference.

---

## Training Pipeline

### 1. Subset Training

Train Pix2Pix on a small subset to stabilize GAN training.

```bash
python train_lab.py
```

Output:

```
checkpoints_subset/
└── G_epoch_030.pt
```

Purpose:

* Fast convergence
* Learn basic color priors
* Stable initialization for later stages

---

### 2. LAB Caching (Optional but Recommended)

RGB → LAB conversion is expensive when repeated every epoch.
LAB caching precomputes this once.

```bash
python cache_lab.py
```

This creates cached LAB tensors on disk.

Benefits:

* Faster training
* Lower CPU overhead
* More stable data loading on Windows

After caching, training uses:

* `dataset_lab_cache.py`

---

### 3. Full Dataset Fine-tuning

Fine-tune the generator on the **entire COCO train2017 dataset**.

```bash
python train_lab_finetune.py
```

Characteristics:

* Initialized from subset checkpoint
* Lower learning rate
* Losses: GAN + L1

Output:

```
checkpoints_finetune/
└── G_finetune_epoch_04.pt
```

---

### 4. Perceptual Fine-tuning with VGG Loss

Add a **VGG16-based perceptual loss** to improve semantic realism.

```bash
python train_vgg.py
```

Purpose:

* Improve texture realism
* Reduce unnatural color artifacts
* Encourage semantic consistency

Output:

```
checkpoints_finetune/
└── G_ft_fromE4_vgg_epoch_01.pt
```

---

## Inference

Colorize a single image or a folder:

```bash
python infer_lab.py <image_or_folder> <checkpoint>
```

Example:

```bash
python infer_lab.py test_images/ checkpoints_finetune/G_ft_fromE4_vgg_epoch_01.pt
```

Inference steps:

1. Convert grayscale image to LAB
2. Normalize L channel
3. Predict ab channels
4. Reconstruct LAB
5. Convert back to RGB

---

## Qualitative Evaluation

Quantitative metrics alone cannot fully evaluate colorization quality.
Human visual inspection is therefore essential.

### Visualization format

```
Grayscale | Ground Truth | Generated (Subset) | Generated (Finetune) | Generated (Finetune+VGG)
```

Utilities used:

* `utils/fixed_samples.py` – select fixed images
* `utils/visualization.py` – generate comparison figures

Qualitative analysis focuses on:

* Color plausibility
* Semantic correctness
* Smoothness of large regions
* Texture realism
* Color bleeding and artifacts

---

## Quantitative Evaluation

Evaluation is performed on the **first 10,000 images of COCO test2017**.

```bash
python calculate_metrics.py
```

Metrics reported:

* **PSNR**
* **SSIM**
* **MS-SSIM**
* **Colorfulness**
* **FID**

---

## Key Observations

* Subset training provides a stable starting point
* Full-dataset fine-tuning significantly improves generalization
* VGG perceptual loss improves visual realism
* LAB color space simplifies the learning problem
* Qualitative evaluation is crucial for fair assessment

---

## Notes

* Pixel-wise metrics (PSNR, SSIM) do not fully reflect perceptual quality
* FID correlates better with human judgment
* Colorization is inherently ambiguous; multiple plausible outputs may exist
