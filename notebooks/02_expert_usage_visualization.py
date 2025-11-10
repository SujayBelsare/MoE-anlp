"""
Script to visualize expert usage over time
Can be run as standalone or converted to notebook
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_expert_usage_history(log_file: str):
    """Load expert usage history from JSON log file"""
    with open(log_file, 'r') as f:
        history = json.load(f)
    return history


def plot_expert_usage_over_time(history, save_path=None):
    """
    Plot expert usage over training steps
    
    Args:
        history: List of expert usage dictionaries
        save_path: Optional path to save figure
    """
    # Extract data
    steps = [h['step'] for h in history]
    epochs = [h['epoch'] for h in history]
    
    # Get encoder and decoder expert usage
    encoder_usage = [h['usage']['encoder'] for h in history]
    decoder_usage = [h['usage']['decoder'] for h in history]
    
    # Number of layers
    n_encoder_layers = len(encoder_usage[0]) if encoder_usage[0] else 0
    n_decoder_layers = len(decoder_usage[0]) if decoder_usage[0] else 0
    
    # Create figure
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    
    # Plot encoder expert usage
    if n_encoder_layers > 0:
        ax = axes[0]
        for layer_idx in range(n_encoder_layers):
            layer_usage = np.array([usage[layer_idx] for usage in encoder_usage])
            num_experts = layer_usage.shape[1]
            
            for expert_idx in range(num_experts):
                ax.plot(steps, layer_usage[:, expert_idx], 
                       label=f'Layer {layer_idx} Expert {expert_idx}',
                       alpha=0.7)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Expert Usage Count')
        ax.set_title('Encoder Expert Usage Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
    
    # Plot decoder expert usage
    if n_decoder_layers > 0:
        ax = axes[1]
        for layer_idx in range(n_decoder_layers):
            layer_usage = np.array([usage[layer_idx] for usage in decoder_usage])
            num_experts = layer_usage.shape[1]
            
            for expert_idx in range(num_experts):
                ax.plot(steps, layer_usage[:, expert_idx],
                       label=f'Layer {layer_idx} Expert {expert_idx}',
                       alpha=0.7)
        
        ax.set_xlabel('Training Step')
        ax.set_ylabel('Expert Usage Count')
        ax.set_title('Decoder Expert Usage Over Time')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', ncol=2)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_expert_distribution_heatmap(history, timestep_idx=-1, save_path=None):
    """
    Plot heatmap of expert usage distribution at a specific timestep
    
    Args:
        history: List of expert usage dictionaries
        timestep_idx: Index of timestep to visualize (-1 for last)
        save_path: Optional path to save figure
    """
    usage = history[timestep_idx]['usage']
    step = history[timestep_idx]['step']
    epoch = history[timestep_idx]['epoch']
    
    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Encoder heatmap
    if usage['encoder']:
        encoder_data = np.array(usage['encoder'])
        sns.heatmap(encoder_data.T, annot=True, fmt='.0f', cmap='YlOrRd',
                   ax=axes[0], cbar_kws={'label': 'Usage Count'})
        axes[0].set_xlabel('Layer')
        axes[0].set_ylabel('Expert')
        axes[0].set_title(f'Encoder Expert Usage (Step {step}, Epoch {epoch})')
    
    # Decoder heatmap
    if usage['decoder']:
        decoder_data = np.array(usage['decoder'])
        sns.heatmap(decoder_data.T, annot=True, fmt='.0f', cmap='YlGnBu',
                   ax=axes[1], cbar_kws={'label': 'Usage Count'})
        axes[1].set_xlabel('Layer')
        axes[1].set_ylabel('Expert')
        axes[1].set_title(f'Decoder Expert Usage (Step {step}, Epoch {epoch})')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_expert_balance(history, save_path=None):
    """
    Plot expert load balance metrics over time
    
    Args:
        history: List of expert usage dictionaries
        save_path: Optional path to save figure
    """
    steps = [h['step'] for h in history]
    
    # Calculate balance metrics for each timestep
    encoder_cv = []  # Coefficient of variation
    decoder_cv = []
    
    for h in history:
        # Encoder balance
        if h['usage']['encoder']:
            enc_data = np.array(h['usage']['encoder'])
            # Average usage across layers
            avg_usage = enc_data.mean(axis=0)
            cv = np.std(avg_usage) / (np.mean(avg_usage) + 1e-10)
            encoder_cv.append(cv)
        
        # Decoder balance
        if h['usage']['decoder']:
            dec_data = np.array(h['usage']['decoder'])
            avg_usage = dec_data.mean(axis=0)
            cv = np.std(avg_usage) / (np.mean(avg_usage) + 1e-10)
            decoder_cv.append(cv)
    
    # Plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    if encoder_cv:
        ax.plot(steps, encoder_cv, label='Encoder', marker='o', alpha=0.7)
    if decoder_cv:
        ax.plot(steps, decoder_cv, label='Decoder', marker='s', alpha=0.7)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Coefficient of Variation (lower = more balanced)')
    ax.set_title('Expert Load Balance Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def plot_expert_specialization(history, layer_idx=0, save_path=None):
    """
    Plot how expert specialization evolves over time
    
    Args:
        history: List of expert usage dictionaries
        layer_idx: Which layer to visualize
        save_path: Optional path to save figure
    """
    steps = [h['step'] for h in history]
    epochs = [h['epoch'] for h in history]
    
    # Extract encoder usage for specific layer
    encoder_usage = []
    for h in history:
        if h['usage']['encoder'] and len(h['usage']['encoder']) > layer_idx:
            encoder_usage.append(h['usage']['encoder'][layer_idx])
    
    if not encoder_usage:
        print(f"No data for layer {layer_idx}")
        return
    
    encoder_usage = np.array(encoder_usage)
    num_experts = encoder_usage.shape[1]
    
    # Create stacked area plot
    fig, ax = plt.subplots(figsize=(14, 7))
    
    ax.stackplot(steps[:len(encoder_usage)], 
                 *[encoder_usage[:, i] for i in range(num_experts)],
                 labels=[f'Expert {i}' for i in range(num_experts)],
                 alpha=0.8)
    
    ax.set_xlabel('Training Step')
    ax.set_ylabel('Cumulative Expert Usage')
    ax.set_title(f'Expert Specialization Over Time (Encoder Layer {layer_idx})')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def generate_expert_report(history, output_file='expert_analysis_report.txt'):
    """
    Generate a text report of expert usage statistics
    
    Args:
        history: List of expert usage dictionaries
        output_file: Path to save report
    """
    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("EXPERT USAGE ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")
        
        # Final timestep statistics
        final_usage = history[-1]['usage']
        final_step = history[-1]['step']
        final_epoch = history[-1]['epoch']
        
        f.write(f"Analysis at Step {final_step} (Epoch {final_epoch})\n")
        f.write("-"*80 + "\n\n")
        
        # Encoder statistics
        if final_usage['encoder']:
            f.write("ENCODER LAYERS:\n")
            encoder_data = np.array(final_usage['encoder'])
            
            for layer_idx in range(len(encoder_data)):
                layer_usage = encoder_data[layer_idx]
                total_usage = layer_usage.sum()
                
                f.write(f"\n  Layer {layer_idx}:\n")
                f.write(f"    Total usage: {total_usage:.0f}\n")
                f.write(f"    Mean usage per expert: {layer_usage.mean():.2f}\n")
                f.write(f"    Std dev: {layer_usage.std():.2f}\n")
                f.write(f"    Min/Max: {layer_usage.min():.0f} / {layer_usage.max():.0f}\n")
                f.write(f"    CV: {layer_usage.std() / (layer_usage.mean() + 1e-10):.4f}\n")
                
                # Most and least used experts
                most_used = layer_usage.argmax()
                least_used = layer_usage.argmin()
                f.write(f"    Most used expert: {most_used} ({layer_usage[most_used]:.0f} times)\n")
                f.write(f"    Least used expert: {least_used} ({layer_usage[least_used]:.0f} times)\n")
        
        # Decoder statistics
        if final_usage['decoder']:
            f.write("\n\nDECODER LAYERS:\n")
            decoder_data = np.array(final_usage['decoder'])
            
            for layer_idx in range(len(decoder_data)):
                layer_usage = decoder_data[layer_idx]
                total_usage = layer_usage.sum()
                
                f.write(f"\n  Layer {layer_idx}:\n")
                f.write(f"    Total usage: {total_usage:.0f}\n")
                f.write(f"    Mean usage per expert: {layer_usage.mean():.2f}\n")
                f.write(f"    Std dev: {layer_usage.std():.2f}\n")
                f.write(f"    Min/Max: {layer_usage.min():.0f} / {layer_usage.max():.0f}\n")
                f.write(f"    CV: {layer_usage.std() / (layer_usage.mean() + 1e-10):.4f}\n")
                
                most_used = layer_usage.argmax()
                least_used = layer_usage.argmin()
                f.write(f"    Most used expert: {most_used} ({layer_usage[most_used]:.0f} times)\n")
                f.write(f"    Least used expert: {least_used} ({layer_usage[least_used]:.0f} times)\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("END OF REPORT\n")
        f.write("="*80 + "\n")
    
    print(f"Report saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Visualize expert usage")
    parser.add_argument("--log_file", type=str, required=True,
                       help="Path to expert usage history JSON file")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations",
                       help="Directory to save visualizations")
    parser.add_argument("--all", action="store_true",
                       help="Generate all visualizations")
    
    args = parser.parse_args()
    
    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    # Load history
    print(f"Loading expert usage history from {args.log_file}")
    history = load_expert_usage_history(args.log_file)
    print(f"Loaded {len(history)} timesteps")
    
    if args.all or True:  # Always generate all by default
        # Plot 1: Expert usage over time
        print("\nGenerating expert usage over time plot...")
        plot_expert_usage_over_time(
            history,
            save_path=f"{args.output_dir}/expert_usage_over_time.png"
        )
        
        # Plot 2: Heatmap at final timestep
        print("\nGenerating expert usage heatmap...")
        plot_expert_distribution_heatmap(
            history,
            timestep_idx=-1,
            save_path=f"{args.output_dir}/expert_usage_heatmap.png"
        )
        
        # Plot 3: Load balance metrics
        print("\nGenerating load balance plot...")
        plot_expert_balance(
            history,
            save_path=f"{args.output_dir}/expert_balance.png"
        )
        
        # Plot 4: Expert specialization
        print("\nGenerating expert specialization plot...")
        plot_expert_specialization(
            history,
            layer_idx=0,
            save_path=f"{args.output_dir}/expert_specialization.png"
        )
        
        # Generate text report
        print("\nGenerating analysis report...")
        generate_expert_report(
            history,
            output_file=f"{args.output_dir}/expert_analysis_report.txt"
        )
    
    print("\nVisualization complete!")
    print(f"All outputs saved to {args.output_dir}")


if __name__ == "__main__":
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    main()