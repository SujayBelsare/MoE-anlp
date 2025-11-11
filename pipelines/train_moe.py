import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from accelerate import Accelerator
from tqdm import tqdm
import os
import json
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transformer import MoETransformer
from pipelines.data_loader import get_data_loader
from typing import Dict
import matplotlib.pyplot as plt
import numpy as np


class MoETrainer:
    """Trainer for MoE Transformer"""
    
    def __init__(self, config: Dict, router_type: str, use_load_balancer: bool = False):
        self.config = config
        self.router_type = router_type
        self.use_load_balancer = use_load_balancer
        
        # Initialize accelerator with DDP kwargs to handle unused parameters in MoE
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        
        self.accelerator = Accelerator(
            mixed_precision='fp16' if config['training']['fp16'] else 'no',
            gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
            kwargs_handlers=[ddp_kwargs],
        )
        
        # Setup model name
        lb_suffix = "_with_lb" if use_load_balancer else "_no_lb"
        self.model_name = f"moe_{router_type}{lb_suffix}"
        
        # Initialize tokenizer
        print("Initializing tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize model
        print(f"Initializing MoE Transformer with {router_type} routing...")
        self.model = MoETransformer(
            vocab_size=config['model']['vocab_size'],
            d_model=config['model']['d_model'],
            nhead=config['model']['nhead'],
            num_encoder_layers=config['model']['num_encoder_layers'],
            num_decoder_layers=config['model']['num_decoder_layers'],
            num_experts=config['model']['num_experts'],
            expert_hidden_dim=config['model']['expert_hidden_dim'],
            top_k=config['model']['top_k'],
            router_type=router_type,
            dropout=config['model']['dropout'],
            max_seq_length=config['model']['max_seq_length'],
            use_load_balancer=use_load_balancer,
            pad_token_id=self.tokenizer.pad_token_id,
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params:,}")
        print(f"Trainable parameters: {trainable_params:,}")
        
        # Initialize optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['training']['learning_rate'],
            weight_decay=config['training']['weight_decay'],
        )
        
        # Load data
        print("Loading data...")
        self.train_loader = get_data_loader(
            self.tokenizer,
            config['training']['batch_size'],
            'train',
            config['data']['max_input_length'],
            config['data']['max_target_length'],
            config['data']['train_samples'],
            config['hardware']['num_workers'],
        )
        
        # Use evaluation batch_size for validation loader
        eval_batch_size = config.get('evaluation', {}).get('batch_size', config['training']['batch_size'])
        self.val_loader = get_data_loader(
            self.tokenizer,
            eval_batch_size,
            'validation',
            config['data']['max_input_length'],
            config['data']['max_target_length'],
            config['data']['val_samples'],
            config['hardware']['num_workers'],
        )
        
        # Learning rate scheduler
        num_training_steps = len(self.train_loader) * config['training']['num_epochs']
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=config['training']['learning_rate'],
            total_steps=num_training_steps,
            pct_start=config['training']['warmup_steps'] / num_training_steps,
        )
        
        # Prepare with accelerator
        self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler = \
            self.accelerator.prepare(
                self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
            )
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.tokenizer.pad_token_id)
        
        # Tracking
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.training_history = []
        self.expert_usage_history = []
    
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_lm_loss = 0
        total_aux_loss = 0
        
        progress_bar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}",
            disable=not self.accelerator.is_local_main_process,
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Prepare inputs
            src = batch['input_ids']
            tgt = batch['labels']
            src_key_padding_mask = (src == self.tokenizer.pad_token_id)
            tgt_key_padding_mask = (tgt == self.tokenizer.pad_token_id)
            
            # Prepare decoder input (shift right)
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            tgt_key_padding_mask = tgt_key_padding_mask[:, :-1]
            
            # Create causal mask for decoder
            tgt_mask = self.model.module.generate_square_subsequent_mask(
                tgt_input.size(1)
            ).to(src.device) if hasattr(self.model, 'module') else \
                self.model.generate_square_subsequent_mask(
                tgt_input.size(1)
            ).to(src.device)
            
            # Forward pass
            logits, aux_loss = self.model(
                src, tgt_input,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                tgt_mask=tgt_mask,
            )
            
            # Compute language modeling loss
            lm_loss = self.criterion(
                logits.reshape(-1, logits.size(-1)),
                tgt_output.reshape(-1)
            )
            
            # Total loss
            loss = lm_loss
            if aux_loss is not None and self.use_load_balancer:
                aux_loss_scaled = self.config['model']['load_balance_loss_coef'] * aux_loss
                loss = loss + aux_loss_scaled
                total_aux_loss += aux_loss_scaled.item()
            
            # Backward pass
            self.accelerator.backward(loss)
            
            if (batch_idx + 1) % self.config['training']['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                self.accelerator.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['max_grad_norm']
                )
                
                # Optimizer step
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                
                self.global_step += 1
                
                # Logging (only after optimizer step)
                if self.global_step % self.config['training']['logging_steps'] == 0:
                    self.training_history.append({
                        'step': self.global_step,
                        'epoch': epoch,
                        'loss': loss.item(),
                        'lm_loss': lm_loss.item(),
                        'aux_loss': aux_loss.item() if aux_loss is not None else 0.0,
                        'lr': self.scheduler.get_last_lr()[0],
                    })
                
                # Save checkpoint (only after optimizer step, skip step 0)
                if self.global_step > 0 and self.global_step % self.config['training']['save_steps'] == 0:
                    self.save_checkpoint(f"step_{self.global_step}")
                
                # Evaluation (only after optimizer step, skip step 0)
                if self.global_step > 0 and self.global_step % self.config['training']['eval_steps'] == 0:
                    val_loss = self.evaluate()
                    self.model.train()
                    
                    if val_loss < self.best_val_loss:
                        self.best_val_loss = val_loss
                        self.save_checkpoint("best")
            
            # Track losses
            total_loss += loss.item()
            total_lm_loss += lm_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'lm_loss': f"{lm_loss.item():.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}",
            })
        
        avg_loss = total_loss / len(self.train_loader)
        avg_lm_loss = total_lm_loss / len(self.train_loader)
        
        return avg_loss, avg_lm_loss
    
    def evaluate(self):
        """Evaluate on validation set"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Evaluating", 
                            disable=not self.accelerator.is_local_main_process):
                src = batch['input_ids']
                tgt = batch['labels']
                src_key_padding_mask = (src == self.tokenizer.pad_token_id)
                
                tgt_input = tgt[:, :-1]
                tgt_output = tgt[:, 1:]
                tgt_key_padding_mask = (tgt[:, :-1] == self.tokenizer.pad_token_id)
                
                # Get the unwrapped model for accessing methods
                unwrapped_model = self.accelerator.unwrap_model(self.model)
                
                tgt_mask = unwrapped_model.generate_square_subsequent_mask(
                    tgt_input.size(1)
                ).to(src.device)
                
                logits, _ = self.model(
                    src, tgt_input,
                    src_key_padding_mask=src_key_padding_mask,
                    tgt_key_padding_mask=tgt_key_padding_mask,
                    tgt_mask=tgt_mask,
                )
                
                loss = self.criterion(
                    logits.reshape(-1, logits.size(-1)),
                    tgt_output.reshape(-1)
                )
                
                # Gather losses from all processes
                total_loss += self.accelerator.gather(loss).mean().item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        
        # Only print on main process
        if self.accelerator.is_local_main_process:
            print(f"Validation Loss: {avg_loss:.4f}")
        
        return avg_loss
    
    def train(self):
        """Main training loop"""
        print(f"\nStarting training for {self.config['training']['num_epochs']} epochs...")
        
        for epoch in range(1, self.config['training']['num_epochs'] + 1):
            train_loss, lm_loss = self.train_epoch(epoch)
            
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  LM Loss: {lm_loss:.4f}")
            
            # Track expert usage
            self.track_expert_usage(epoch)
            
            # Evaluate
            val_loss = self.evaluate()
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.save_checkpoint("best")
        
        # Save final model
        self.save_checkpoint("final")
        
        # Visualize expert usage
        self.visualize_expert_usage()
        
        print("\nTraining completed!")
    
    def track_expert_usage(self, epoch: int):
        """Track expert usage statistics"""
        if hasattr(self.model, 'module'):
            stats = self.model.module.get_all_expert_usage_stats()
        else:
            stats = self.model.get_all_expert_usage_stats()
        
        self.expert_usage_history.append({
            'epoch': epoch,
            'step': self.global_step,
            'stats': stats,
        })
    
    def visualize_expert_usage(self):
        """Visualize expert usage over time"""
        if not self.expert_usage_history:
            return
        
        output_dir = os.path.join(
            self.config['output']['visualizations_dir'],
            self.model_name
        )
        os.makedirs(output_dir, exist_ok=True)
        
        # Plot encoder expert usage
        num_encoder_layers = self.config['model']['num_encoder_layers']
        fig, axes = plt.subplots(num_encoder_layers, 1, figsize=(12, 4 * num_encoder_layers))
        
        if num_encoder_layers == 1:
            axes = [axes]
        
        for layer_idx in range(num_encoder_layers):
            usage_over_time = []
            epochs = []
            
            for record in self.expert_usage_history:
                epochs.append(record['epoch'])
                layer_usage = record['stats']['encoder'][layer_idx]['usage']
                usage_over_time.append(layer_usage)
            
            usage_array = np.array(usage_over_time)
            
            for expert_idx in range(self.config['model']['num_experts']):
                axes[layer_idx].plot(
                    epochs,
                    usage_array[:, expert_idx],
                    label=f'Expert {expert_idx}',
                    marker='o'
                )
            
            axes[layer_idx].set_xlabel('Epoch')
            axes[layer_idx].set_ylabel('Usage Proportion')
            axes[layer_idx].set_title(f'Encoder Layer {layer_idx} - Expert Usage Over Time')
            axes[layer_idx].legend()
            axes[layer_idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'encoder_expert_usage.png'), dpi=300)
        plt.close()
        
        print(f"Expert usage visualizations saved to {output_dir}")
    
    def save_checkpoint(self, name: str):
        """Save model checkpoint"""
        output_dir = os.path.join(self.config['output']['model_dir'], self.model_name)
        os.makedirs(output_dir, exist_ok=True)
        
        checkpoint_dir = os.path.join(output_dir, name)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Unwrap model if using accelerator
        unwrapped_model = self.accelerator.unwrap_model(self.model)
        
        # Save model
        torch.save({
            'model_state_dict': unwrapped_model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'best_val_loss': self.best_val_loss,
            'config': self.config,
            'router_type': self.router_type,
            'use_load_balancer': self.use_load_balancer,
        }, os.path.join(checkpoint_dir, 'checkpoint.pt'))
        
        # Save training history
        with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
            json.dump(self.training_history, f, indent=2)
        
        # Save expert usage history
        with open(os.path.join(output_dir, 'expert_usage_history.json'), 'w') as f:
            json.dump(self.expert_usage_history, f, indent=2)
        
        print(f"Checkpoint saved to {checkpoint_dir}")


def train_moe_model(config: Dict, router_type: str, use_load_balancer: bool = False):
    """Train a single MoE model"""
    trainer = MoETrainer(config, router_type, use_load_balancer)
    trainer.train()