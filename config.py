# config_subset.py
import torch

# ==========================
# DEVICE
# ==========================
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ==========================
# DATA
# ==========================
DATA_ROOT = "coco_subset_colorization"
IMG_SIZE = 256

# ==========================
# TRAINING
# ==========================
EPOCHS = 30
BATCH_SIZE = 16        
LAMBDA_L1 = 100

NUM_WORKERS = 0         
PIN_MEMORY = True

# ==========================
# SPEED / LOGGING
# ==========================
VIS_EVERY = 2

# ==========================
# OUTPUT
# ==========================
CHECKPOINT_DIR = "checkpoints_subset"
VIS_DIR = "visuals"
LOG_DIR = "logs"
