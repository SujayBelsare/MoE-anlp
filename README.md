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
