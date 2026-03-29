from pathlib import Path

import torch

DATA_ROOT = Path("dataset")
CKPT_DIR = Path("checkpoints")
IMG_SIZE = 28
BATCH_SIZE = 64
EPOCHS = 30
LR = 1e-3
HIDDEN = [512, 256]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
