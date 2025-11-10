"""
Main entry point for the project
"""

from configs import load_config
from pipelines import run_baselines, train_moe, evaluate


def main():
    # Load configuration
    config = load_config()
    
    # Run in specified mode from config
    mode = config.get('mode', 'train')
    
    if mode == "baseline":
        print("Running baseline experiments...")
        run_baselines.main()
    elif mode == "train":
        print("Training MoE model...")
        train_moe.main()
    elif mode == "evaluate":
        print("Evaluating models...")
        evaluate.main()


if __name__ == "__main__":
    main()
