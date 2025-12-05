import numpy as np
import torch

# -------------------------
# Reconstruction mode
# -------------------------
# "nb"  → Negative Binomial（for count data）
# "mse" → MSE（for PCA or normalized data）
RECON_MODE = "nb"   # "nb" / "mse"

# -------------------------
# Model dimensions
# -------------------------
LATENT_DIM = 64
SPECIES_EMB_DIM = 12
HIDDEN_DIMS = [256, 128]

# -------------------------
# Training hyper-parameters
# -------------------------
BATCH_SIZE = 128

EPOCHS_PHASE1 = 5      # Phase1 epochs（two-phase only）
EPOCHS_PHASE2 = 5      # Phase2 epochs（two-phase only）
EPOCHS_ONEPHASE = 5    # one-phase epochs

LR = 5e-4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

PRINT_EVERY = 1        # print loss every n epoches

# -------------------------
# Loss weights
# -------------------------
if RECON_MODE == "nb":
    RECON_W = 1.0
else:
    RECON_W = 5.0

CE_MAX_W = 2.0
CTR_MAX_W = 1.0
KL_MAX_W  = 1.0
MMD_MAX_W = 0.2
SPECIES_MAX_W = 10

RUN_SEEDS = list(range(20))

# -------------------------s
# Data path
# -------------------------
NPZ_PATH = "../data/frog_zeb_processed_cnt.npz"
