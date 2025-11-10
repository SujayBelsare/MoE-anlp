"""
Configuration file for the project
"""

import torch

# Dataset configuration
DATASET_NAME = "EdinburghNLP/xsum"
MAX_SOURCE_LENGTH = 512
MAX_TARGET_LENGTH = 64

# Model configuration
D_MODEL = 512
N_HEADS = 8
N_LAYERS = 6
D_FF = 2048
DROPOUT = 0.1

# MoE configuration
NUM_EXPERTS = 8
TOP_K = 2
LOAD_BALANCER_WEIGHT = 0.01

# Training configuration
BATCH_SIZE = 16
LEARNING_RATE = 5e-5
NUM_EPOCHS = 10
WARMUP_STEPS = 1000
GRADIENT_ACCUMULATION_STEPS = 4
MAX_GRAD_NORM = 1.0

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Paths
CHECKPOINT_DIR = "checkpoints"
OUTPUT_DIR = "outputs"
LOG_DIR = "logs"

# HuggingFace Hub
HF_USERNAME = "your-username"  # Update with your HF username

# Baseline models
BART_MODEL = "facebook/bart-large-xsum"
T5_MODELS = ["google-t5/t5-base", "google-t5/t5-large", "google/pegasus-large"]
INSTRUCTION_MODELS = [
    "meta-llama/Llama-3.2-1B-Instruct",
    "meta-llama/Llama-3.2-3B-Instruct",
    "Qwen/Qwen2.5-3B-Instruct"
]

# Evaluation
NUM_BEAMS = 4
LENGTH_PENALTY = 2.0
EARLY_STOPPING = True
