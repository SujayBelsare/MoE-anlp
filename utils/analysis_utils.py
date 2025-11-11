"""
Utility functions for analysis and visualization
"""

import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from typing import Dict, List


def compare_all_models(results_dir: str, output_path: str = "model_comparison.png"):
    """
    Create comparison plots for all models
    
    Args:
        results_dir: Directory containing all results
        output_path: Path to save comparison plot
    """
    # Load all metrics
    models = []
    metrics_data = []
    
    model_dirs = [
        'bart',
        'encoder_decoder',
        'moe_hash_no_lb',
        'moe_top_k_no_lb',
        'moe_hash_with_lb',
        'moe_top_k_with_lb',
    ]
    
    for model_dir in model_dirs:
        metrics_file = os.path.join(results_dir, model_dir, 'metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
                models.append(model_dir)
                metrics_data.append(data)
    
    if not metrics_data:
        print("No metrics files found!")
        return
    
    # Extract metrics
    rouge1_scores = [m['lexical_metrics']['rouge1']['mean'] for m in metrics_data]
    rouge2_scores = [m['lexical_metrics']['rouge2']['mean'] for m in metrics_data]
    rougeL_scores = [m['lexical_metrics']['rougeL']['mean'] for m in metrics_data]
    bleu_scores = [m['lexical_metrics']['bleu']['mean'] for m in metrics_data]
    bert_scores = [m['embedding_metrics']['f1']['mean'] for m in metrics_data]
    compression_ratios = [m['document_metrics']['compression_ratio']['mean'] for m in metrics_data]
    extractiveness = [m['document_metrics']['extractiveness']['mean'] for m in metrics_data]
    
    # Create comparison plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Model Comparison - All Metrics', fontsize=16, fontweight='bold')
    
    # ROUGE-1
    axes[0, 0].bar(range(len(models)), rouge1_scores, color='steelblue')
    axes[0, 0].set_xticks(range(len(models)))
    axes[0, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 0].set_ylabel('Score')
    axes[0, 0].set_title('ROUGE-1')
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # ROUGE-2
    axes[0, 1].bar(range(len(models)), rouge2_scores, color='coral')
    axes[0, 1].set_xticks(range(len(models)))
    axes[0, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('ROUGE-2')
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # ROUGE-L
    axes[0, 2].bar(range(len(models)), rougeL_scores, color='mediumseagreen')
    axes[0, 2].set_xticks(range(len(models)))
    axes[0, 2].set_xticklabels(models, rotation=45, ha='right')
    axes[0, 2].set_ylabel('Score')
    axes[0, 2].set_title('ROUGE-L')
    axes[0, 2].grid(axis='y', alpha=0.3)
    
    # BLEU
    axes[1, 0].bar(range(len(models)), bleu_scores, color='mediumpurple')
    axes[1, 0].set_xticks(range(len(models)))
    axes[1, 0].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 0].set_ylabel('Score')
    axes[1, 0].set_title('BLEU')
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # BERTScore
    axes[1, 1].bar(range(len(models)), bert_scores, color='gold')
    axes[1, 1].set_xticks(range(len(models)))
    axes[1, 1].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 1].set_ylabel('Score')
    axes[1, 1].set_title('BERTScore F1')
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    # Compression & Extractiveness
    x = np.arange(len(models))
    width = 0.35
    axes[1, 2].bar(x - width/2, compression_ratios, width, label='Compression', color='lightcoral')
    axes[1, 2].bar(x + width/2, extractiveness, width, label='Extractiveness', color='lightblue')
    axes[1, 2].set_xticks(range(len(models)))
    axes[1, 2].set_xticklabels(models, rotation=45, ha='right')
    axes[1, 2].set_ylabel('Score')
    axes[1, 2].set_title('Document Metrics')
    axes[1, 2].legend()
    axes[1, 2].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Comparison plot saved to {output_path}")
    plt.close()


def create_metrics_table(results_dir: str, output_path: str = "metrics_table.csv"):
    """
    Create a comprehensive metrics table
    
    Args:
        results_dir: Directory containing all results
        output_path: Path to save CSV table
    """
    model_dirs = [
        'bart',
        'encoder_decoder',
        'moe_hash_no_lb',
        'moe_top_k_no_lb',
        'moe_hash_with_lb',
        'moe_top_k_with_lb',
    ]
    
    data = []
    
    for model_dir in model_dirs:
        metrics_file = os.path.join(results_dir, model_dir, 'metrics.json')
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                metrics = json.load(f)
                
                row = {
                    'Model': model_dir,
                    'ROUGE-1': f"{metrics['lexical_metrics']['rouge1']['mean']:.4f} ± {metrics['lexical_metrics']['rouge1']['std']:.4f}",
                    'ROUGE-2': f"{metrics['lexical_metrics']['rouge2']['mean']:.4f} ± {metrics['lexical_metrics']['rouge2']['std']:.4f}",
                    'ROUGE-L': f"{metrics['lexical_metrics']['rougeL']['mean']:.4f} ± {metrics['lexical_metrics']['rougeL']['std']:.4f}",
                    'BLEU': f"{metrics['lexical_metrics']['bleu']['mean']:.4f} ± {metrics['lexical_metrics']['bleu']['std']:.4f}",
                    'BERTScore F1': f"{metrics['embedding_metrics']['f1']['mean']:.4f} ± {metrics['embedding_metrics']['f1']['std']:.4f}",
                    'Compression': f"{metrics['document_metrics']['compression_ratio']['mean']:.4f}",
                    'Extractiveness': f"{metrics['document_metrics']['extractiveness']['mean']:.4f}",
                }
                data.append(row)
    
    if data:
        df = pd.DataFrame(data)
        df.to_csv(output_path, index=False)
        print(f"\nMetrics table saved to {output_path}")
        print("\n" + df.to_string(index=False))
    else:
        print("No metrics found!")


def plot_training_curves(model_dir: str, output_path: str = None):
    """
    Plot training curves from training history
    
    Args:
        model_dir: Directory containing training_history.json
        output_path: Path to save plot (None = display only)
    """
    history_file = os.path.join(model_dir, 'training_history.json')
    
    if not os.path.exists(history_file):
        print(f"Training history not found: {history_file}")
        return
    
    with open(history_file, 'r') as f:
        history = json.load(f)
    
    steps = [h['step'] for h in history]
    losses = [h['loss'] for h in history]
    lm_losses = [h['lm_loss'] for h in history]
    aux_losses = [h['aux_loss'] for h in history]
    lrs = [h['lr'] for h in history]
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'Training Curves - {os.path.basename(model_dir)}', 
                 fontsize=14, fontweight='bold')
    
    # Total loss
    axes[0, 0].plot(steps, losses, label='Total Loss', color='darkblue')
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].grid(alpha=0.3)
    axes[0, 0].legend()
    
    # LM loss
    axes[0, 1].plot(steps, lm_losses, label='LM Loss', color='darkgreen')
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('Language Modeling Loss')
    axes[0, 1].grid(alpha=0.3)
    axes[0, 1].legend()
    
    # Auxiliary loss
    if any(aux > 0 for aux in aux_losses):
        axes[1, 0].plot(steps, aux_losses, label='Aux Loss', color='darkred')
        axes[1, 0].set_xlabel('Steps')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].set_title('Auxiliary Loss (Load Balancing)')
        axes[1, 0].grid(alpha=0.3)
        axes[1, 0].legend()
    else:
        axes[1, 0].text(0.5, 0.5, 'No Auxiliary Loss', 
                       ha='center', va='center', fontsize=12)
        axes[1, 0].set_title('Auxiliary Loss')
    
    # Learning rate
    axes[1, 1].plot(steps, lrs, label='Learning Rate', color='darkorange')
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Learning Rate')
    axes[1, 1].set_title('Learning Rate Schedule')
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].ticklabel_format(style='scientific', axis='y', scilimits=(0,0))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Training curves saved to {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_expert_usage(model_dir: str, output_path: str = None):
    """
    Analyze expert usage patterns
    
    Args:
        model_dir: Directory containing expert_usage_history.json
        output_path: Path to save analysis (None = print only)
    """
    usage_file = os.path.join(model_dir, 'expert_usage_history.json')
    
    if not os.path.exists(usage_file):
        print(f"Expert usage history not found: {usage_file}")
        return
    
    with open(usage_file, 'r') as f:
        usage_history = json.load(f)
    
    if not usage_history:
        print("No usage history available")
        return
    
    # Analyze final epoch
    final_stats = usage_history[-1]['stats']
    
    print(f"\nExpert Usage Analysis - {os.path.basename(model_dir)}")
    print("=" * 60)
    
    # Encoder layers
    print("\nEncoder Layers:")
    for layer_stats in final_stats['encoder']:
        layer_idx = layer_stats['layer']
        usage = np.array(layer_stats['usage'])
        print(f"  Layer {layer_idx}:")
        print(f"    Mean usage: {usage.mean():.4f}")
        print(f"    Std usage: {usage.std():.4f}")
        print(f"    Min usage: {usage.min():.4f} (Expert {usage.argmin()})")
        print(f"    Max usage: {usage.max():.4f} (Expert {usage.argmax()})")
        print(f"    Usage balance: {1 - usage.std() / usage.mean():.4f}")
    
    # Decoder layers
    print("\nDecoder Layers:")
    for layer_stats in final_stats['decoder']:
        layer_idx = layer_stats['layer']
        usage = np.array(layer_stats['usage'])
        print(f"  Layer {layer_idx}:")
        print(f"    Mean usage: {usage.mean():.4f}")
        print(f"    Std usage: {usage.std():.4f}")
        print(f"    Min usage: {usage.min():.4f} (Expert {usage.argmin()})")
        print(f"    Max usage: {usage.max():.4f} (Expert {usage.argmax()})")
        print(f"    Usage balance: {1 - usage.std() / usage.mean():.4f}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        results_dir = sys.argv[1]
    else:
        results_dir = "./results"
    
    print("Running comprehensive analysis...")
    
    # Compare all models
    compare_all_models(results_dir)
    
    # Create metrics table
    create_metrics_table(results_dir)
    
    print("\nAnalysis complete!")