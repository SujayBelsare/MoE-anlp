#!/usr/bin/env python3
"""
Main entry point for ANLP Assignment 3 - MoE Transformer for Extreme Summarization
"""

import argparse
import yaml
import os
import torch
import random
import numpy as np

from pipelines.run_baselines import (
    run_bart_inference,
    finetune_encoder_decoder,
    instruction_tune_model,
    inference_instruct_model,
    instruct_base_model,
)
from pipelines.train_moe import train_moe_model
from pipelines.evaluate import evaluate_moe_model, evaluate_baseline


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(config_path: str = "config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_output_directories(config):
    """Create necessary output directories"""
    os.makedirs(config['output']['model_dir'], exist_ok=True)
    os.makedirs(config['output']['results_dir'], exist_ok=True)
    os.makedirs(config['output']['logs_dir'], exist_ok=True)
    os.makedirs(config['output']['visualizations_dir'], exist_ok=True)


def main():
    parser = argparse.ArgumentParser(
        description="ANLP Assignment 3 - MoE Transformer for Extreme Summarization"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        choices=[
            "bart_inference",
            "finetune_encoder_decoder",
            "instruction_tune",
            "instruct_inference",
            "instruct_base_model",
            "train_moe_hash",
            "train_moe_topk",
            "train_moe_hash_lb",
            "train_moe_topk_lb",
            "eval_moe_hash",
            "eval_moe_topk",
            "eval_moe_hash_lb",
            "eval_moe_topk_lb",
            "eval_bart",
            "eval_encoder_decoder",
            "all_baselines",
            "all_moe",
            "all_eval",
        ],
        help="Task to run"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    print("Loading configuration...")
    config = load_config(args.config)
    
    # Set seed
    set_seed(config['training']['seed'])
    
    # Create directories
    create_output_directories(config)
    
    # Execute task
    print(f"\n{'='*70}")
    print(f"Running task: {args.task}")
    print(f"{'='*70}\n")
    
    if args.task == "bart_inference":
        run_bart_inference(config, config['output']['results_dir'])
    
    elif args.task == "finetune_encoder_decoder":
        finetune_encoder_decoder(config, config['output']['results_dir'])
    
    elif args.task == "instruction_tune":
        instruction_tune_model(config, config['output']['results_dir'])
    
    elif args.task == "instruct_inference":
        inference_instruct_model(config, config['output']['results_dir'], model_path="/scratch/sujayb/MoE-anlp/results/instruct_model/checkpoint-4785")
    
    elif args.task == "instruct_base_model":
        instruct_base_model(config, config['output']['results_dir'])

    elif args.task == "train_moe_hash":
        train_moe_model(config, router_type="hash", use_load_balancer=False)
    
    elif args.task == "train_moe_topk":
        train_moe_model(config, router_type="top_k", use_load_balancer=False)
    
    elif args.task == "train_moe_hash_lb":
        train_moe_model(config, router_type="hash", use_load_balancer=True)
    
    elif args.task == "train_moe_topk_lb":
        train_moe_model(config, router_type="top_k", use_load_balancer=True)
    
    elif args.task == "eval_moe_hash":
        evaluate_moe_model(config, router_type="hash", use_load_balancer=False)
    
    elif args.task == "eval_moe_topk":
        evaluate_moe_model(config, router_type="top_k", use_load_balancer=False)
    
    elif args.task == "eval_moe_hash_lb":
        evaluate_moe_model(config, router_type="hash", use_load_balancer=True)
    
    elif args.task == "eval_moe_topk_lb":
        evaluate_moe_model(config, router_type="top_k", use_load_balancer=True)
    
    elif args.task == "eval_bart":
        bart_results_file = os.path.join(config['output']['results_dir'], 'bart_results.json')
        evaluate_baseline(bart_results_file, 'bart', config)
    
    elif args.task == "eval_encoder_decoder":
        ed_results_file = os.path.join(config['output']['results_dir'], 'encoder_decoder_results.json')
        evaluate_baseline(ed_results_file, 'encoder_decoder', config)
    
    elif args.task == "all_baselines":
        print("\n>>> Running all baseline experiments...\n")
        run_bart_inference(config, config['output']['results_dir'])
        finetune_encoder_decoder(config, config['output']['results_dir'])
        instruction_tune_model(config, config['output']['results_dir'])
    
    elif args.task == "all_moe":
        print("\n>>> Training all MoE models...\n")
        train_moe_model(config, router_type="hash", use_load_balancer=False)
        train_moe_model(config, router_type="top_k", use_load_balancer=False)
    
    elif args.task == "all_eval":
        print("\n>>> Evaluating all models...\n")
        
        # Evaluate baselines
        bart_results_file = os.path.join(config['output']['results_dir'], 'bart_results.json')
        if os.path.exists(bart_results_file):
            evaluate_baseline(bart_results_file, 'bart', config)
        
        ed_results_file = os.path.join(config['output']['results_dir'], 'encoder_decoder_results.json')
        if os.path.exists(ed_results_file):
            evaluate_baseline(ed_results_file, 'encoder_decoder', config)
        
        # Evaluate MoE models
        evaluate_moe_model(config, router_type="hash", use_load_balancer=False)
        evaluate_moe_model(config, router_type="top_k", use_load_balancer=False)
    
    print(f"\n{'='*70}")
    print(f"Task '{args.task}' completed successfully!")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    main()