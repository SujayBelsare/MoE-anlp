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
