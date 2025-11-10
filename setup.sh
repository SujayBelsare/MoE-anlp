#!/bin/bash

# Advanced NLP Assignment 3 - Project Setup Script
# This script creates the complete project structure with all required files

echo "=========================================="
echo "Advanced NLP Assignment 3 - Setup Script"
echo "=========================================="
echo ""

# Create main project directory
PROJECT_DIR="anlp_assignment_3"
echo "Creating project directory: $PROJECT_DIR"
mkdir -p $PROJECT_DIR
cd $PROJECT_DIR

# Create directory structure
echo "Creating directory structure..."
mkdir -p models
mkdir -p pipelines
mkdir -p notebooks
mkdir -p configs
mkdir -p checkpoints
mkdir -p outputs
mkdir -p logs

# Create __init__.py files
echo "Creating __init__.py files..."
touch models/__init__.py
touch pipelines/__init__.py

# Create model files
echo "Creating model files..."
touch models/moe_layer.py
touch models/routing.py
touch models/load_balancer.py
touch models/transformer.py
touch models/bonus.py

# Create pipeline files
echo "Creating pipeline files..."
touch pipelines/data_loader.py
touch pipelines/run_baselines.py
touch pipelines/train_moe.py
touch pipelines/evaluate.py

# Create notebook files
echo "Creating notebook files..."
touch notebooks/01_data_exploration.ipynb
touch notebooks/02_expert_usage_visualization.ipynb

# Create main files
echo "Creating main project files..."
touch main.py
touch config.py
touch utils.py

# Create requirements.txt
echo "Creating requirements.txt..."
cat > requirements.txt << 'EOF'
# Core dependencies
torch>=2.0.0
transformers>=4.30.0
datasets>=2.14.0
tokenizers>=0.13.0

# Training and optimization
accelerate>=0.20.0
peft>=0.4.0
bitsandbytes>=0.41.0

# Evaluation metrics
rouge-score>=0.1.2
bert-score>=0.3.13
nltk>=3.8.1
sacrebleu>=2.3.1

# Factual consistency
summac>=0.0.4

# Utilities
numpy>=1.24.0
pandas>=2.0.0
matplotlib>=3.7.0
seaborn>=0.12.0
tqdm>=4.65.0
scikit-learn>=1.3.0

# Logging and experiment tracking
wandb>=0.15.0
tensorboard>=2.13.0

# HuggingFace Hub
huggingface-hub>=0.16.0

# Development
jupyter>=1.0.0
ipython>=8.14.0
black>=23.7.0
EOF

# Create README.md
echo "Creating README.md..."
cat > README.md << 'EOF'
# Advanced NLP Assignment 3 - Mixture of Experts for Extreme Summarization

## Project Overview
Implementation of a Sparse Mixture-of-Experts (MoE) Transformer from scratch for the task of Extreme Summarization on the XSum dataset.

## Setup Instructions


### 2. Dataset
The project uses the `EdinburghNLP/xsum` dataset which will be automatically downloaded when running the scripts.

### 3. Running the Code

#### Baseline Models
```bash
# Run BART inference (Task 1)
python pipelines/run_baselines.py --model bart --task inference

# Finetune encoder-decoder model (Task 2)
python pipelines/run_baselines.py --model t5-base --task finetune

# Instruction tune model (Task 3)
python pipelines/run_baselines.py --model llama-1b --task instruction-tune
```

#### MoE Models
```bash
# Train MoE with Hash Routing
python pipelines/train_moe.py --routing hash --epochs 10

# Train MoE with Top-K Routing
python pipelines/train_moe.py --routing topk --k 2 --epochs 10

# Train with load balancer loss (Bonus 1)
python pipelines/train_moe.py --routing topk --k 2 --use-load-balancer --epochs 10
```

#### Evaluation
```bash
# Run comprehensive evaluation
python pipelines/evaluate.py --model-dir checkpoints/moe_topk
```

## Project Structure
```
anlp_assignment_3/
├── models/              # Model implementations
├── pipelines/           # Training and evaluation scripts
├── notebooks/           # Analysis notebooks
├── configs/             # Configuration files
├── checkpoints/         # Saved model checkpoints
├── outputs/             # Generated summaries
├── logs/                # Training logs
├── main.py              # Main CLI interface
├── requirements.txt     # Dependencies
└── report.pdf           # Final report
```

## Model Checkpoints
All trained models will be pushed to HuggingFace Hub. Links will be provided in the report.

## Authors
[Your Name] - [Your Roll Number]

## Deadline
10th November 2025, 23:59
EOF

# Create config.py
echo "Creating config.py..."
cat > config.py << 'EOF'
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
EOF

# Create utils.py
echo "Creating utils.py..."
cat > utils.py << 'EOF'
"""
Utility functions for the project
"""

import os
import json
import torch
import random
import numpy as np
from pathlib import Path


def set_seed(seed=42):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_dir(directory):
    """Create directory if it doesn't exist"""
    Path(directory).mkdir(parents=True, exist_ok=True)


def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)


def load_checkpoint(model, optimizer, path):
    """Load model checkpoint"""
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['loss']


def save_json(data, path):
    """Save data to JSON file"""
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(path):
    """Load data from JSON file"""
    with open(path, 'r') as f:
        return json.load(f)


def count_parameters(model):
    """Count total and trainable parameters in model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_time(seconds):
    """Format seconds to readable time string"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"
EOF

# Create main.py
echo "Creating main.py..."
cat > main.py << 'EOF'
"""
Main entry point for the project
"""

import argparse
from pipelines import run_baselines, train_moe, evaluate


def main():
    parser = argparse.ArgumentParser(description="Advanced NLP Assignment 3")
    parser.add_argument("--mode", type=str, required=True,
                       choices=["baseline", "train", "evaluate"],
                       help="Mode to run")
    parser.add_argument("--model", type=str, help="Model name")
    parser.add_argument("--routing", type=str, choices=["hash", "topk"],
                       help="Routing algorithm for MoE")
    parser.add_argument("--config", type=str, default="config.py",
                       help="Path to config file")
    
    args = parser.parse_args()
    
    if args.mode == "baseline":
        print("Running baseline experiments...")
        # Add baseline execution logic
    elif args.mode == "train":
        print("Training MoE model...")
        # Add training logic
    elif args.mode == "evaluate":
        print("Evaluating models...")
        # Add evaluation logic


if __name__ == "__main__":
    main()
EOF

# Create .gitignore
echo "Creating .gitignore..."
cat > .gitignore << 'EOF'
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/
ENV/
.venv

# Jupyter Notebook
.ipynb_checkpoints
*.ipynb_checkpoints

# Model checkpoints and outputs
checkpoints/
outputs/
logs/
wandb/

# Data
data/
*.csv
*.json
*.txt

# IDE
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Keep directory structure
!.gitkeep
EOF

# Create placeholder .gitkeep files
echo "Creating .gitkeep files..."
touch checkpoints/.gitkeep
touch outputs/.gitkeep
touch logs/.gitkeep

echo ""
echo "=========================================="
echo "Project setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. cd $PROJECT_DIR"
echo "2. Create and activate virtual environment"
echo "3. pip install -r requirements.txt"
echo "4. Review and update config.py"
echo "5. Start implementing the models!"
echo ""
echo "Happy coding!"