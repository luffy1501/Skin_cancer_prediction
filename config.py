import os
import torch

# Model configuration
MODEL_PATH = "models/saved_model.pth"
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# Data configuration
DATA_DIR = "data"
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")

# Training configuration
BATCH_SIZE = 32
LEARNING_RATE = 0.001
NUM_EPOCHS = 50
IMAGE_SIZE = 224

# Preprocessing parameters
NORMALIZE_MEAN = [0.485, 0.456, 0.406]
NORMALIZE_STD = [0.229, 0.224, 0.225]

# Kaggle dataset
KAGGLE_DATASET = "kmader/skin-cancer-mnist-ham10000"

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
