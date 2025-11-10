"""
Training script for MoE Transformer model
"""

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import get_linear_schedule_with_warmup
from transformers import AutoTokenizer
import os
from tqdm import tqdm
import wandb
from pathlib import Path
import json
import numpy as np

from configs import load_config
from models.transformer import MoETransformer
from pipelines.data_loader import XSumDataModule
from utils import set_seed, ensure_dir, save_checkpoint, count_parameters, format_time
import time

# Load default configuration
config = load_config()


class MoETrainer:
    """Trainer class for MoE Transformer"""
    
    def __init__(
        self,
        model: MoETransformer,
        tokenizer,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        device: str,
        config_dict: dict,
        checkpoint_dir: str = "checkpoints",
        log_dir: str = "logs",
        use_wandb: bool = True
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config_dict
        self.checkpoint_dir = checkpoint_dir
        self.log_dir = log_dir
        self.use_wandb = use_wandb
        
        # Create directories
        ensure_dir(checkpoint_dir)
        ensure_dir(log_dir)
        
        # Move model to device
        self.model.to(device)
        
        # Loss function (ignore padding tokens)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        
        # Expert usage tracking
        self.expert_usage_history = []
        
    def train_epoch(self, epoch: int):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        total_ce_loss = 0
        total_lb_loss = 0
        
        progress_bar = tqdm(self.train_loader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Prepare decoder input (shift labels right)
            decoder_input_ids = torch.cat([
                torch.full((labels.size(0), 1), self.tokenizer.pad_token_id, device=self.device),
                labels[:, :-1]
            ], dim=1)
            
            # Forward pass
            logits, lb_loss = self.model(
                input_ids,
                decoder_input_ids,
                return_load_balancer_loss=True
            )
            
            # Compute cross-entropy loss
            ce_loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            # Total loss
            loss = ce_loss
            if lb_loss is not None:
                loss = loss + lb_loss
                total_lb_loss += lb_loss.item()
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            if self.config.get('max_grad_norm', 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )
            
            # Optimizer step
            if (batch_idx + 1) % self.config.get('gradient_accumulation_steps', 1) == 0:
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            # Update statistics
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'ce_loss': f"{ce_loss.item():.4f}",
                'lb_loss': f"{lb_loss.item() if lb_loss is not None else 0:.4f}",
                'lr': f"{self.scheduler.get_last_lr()[0]:.2e}"
            })
            
            # Log to wandb
            if self.use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss': loss.item(),
                    'train/ce_loss': ce_loss.item(),
                    'train/lb_loss': lb_loss.item() if lb_loss is not None else 0,
                    'train/learning_rate': self.scheduler.get_last_lr()[0],
                    'train/epoch': epoch,
                    'train/step': self.global_step
                })
            
            # Track expert usage periodically
            if batch_idx % 100 == 0:
                usage_stats = self.model.get_expert_usage()
                self.expert_usage_history.append({
                    'step': self.global_step,
                    'epoch': epoch,
                    'usage': usage_stats
                })
        
        # Average losses
        avg_loss = total_loss / len(self.train_loader)
        avg_ce_loss = total_ce_loss / len(self.train_loader)
        avg_lb_loss = total_lb_loss / len(self.train_loader) if total_lb_loss > 0 else 0
        
        return avg_loss, avg_ce_loss, avg_lb_loss
    
    @torch.no_grad()
    def validate(self, epoch: int):
        """Validate the model"""
        self.model.eval()
        total_loss = 0
        total_ce_loss = 0
        total_lb_loss = 0
        
        progress_bar = tqdm(self.val_loader, desc="Validation")
        
        for batch in progress_bar:
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Prepare decoder input
            decoder_input_ids = torch.cat([
                torch.full((labels.size(0), 1), self.tokenizer.pad_token_id, device=self.device),
                labels[:, :-1]
            ], dim=1)
            
            # Forward pass
            logits, lb_loss = self.model(
                input_ids,
                decoder_input_ids,
                return_load_balancer_loss=True
            )
            
            # Compute loss
            ce_loss = self.criterion(
                logits.view(-1, logits.size(-1)),
                labels.view(-1)
            )
            
            loss = ce_loss
            if lb_loss is not None:
                loss = loss + lb_loss
                total_lb_loss += lb_loss.item()
            
            total_loss += loss.item()
            total_ce_loss += ce_loss.item()
            
            progress_bar.set_postfix({'val_loss': f"{loss.item():.4f}"})
        
        # Average losses
        avg_loss = total_loss / len(self.val_loader)
        avg_ce_loss = total_ce_loss / len(self.val_loader)
        avg_lb_loss = total_lb_loss / len(self.val_loader) if total_lb_loss > 0 else 0
        
        return avg_loss, avg_ce_loss, avg_lb_loss
    
    def train(self, num_epochs: int, save_every: int = 1, push_to_hub: bool = False):
        """Main training loop"""
        print(f"\nStarting training for {num_epochs} epochs...")
        print(f"Total parameters: {count_parameters(self.model)[0]:,}")
        print(f"Trainable parameters: {count_parameters(self.model)[1]:,}")
        
        start_time = time.time()
        
        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch
            epoch_start = time.time()
            
            # Train
            train_loss, train_ce, train_lb = self.train_epoch(epoch)
            
            # Validate
            val_loss, val_ce, val_lb = self.validate(epoch)
            
            epoch_time = time.time() - epoch_start
            
            # Log epoch results
            print(f"\nEpoch {epoch}/{num_epochs} - Time: {format_time(epoch_time)}")
            print(f"Train Loss: {train_loss:.4f} (CE: {train_ce:.4f}, LB: {train_lb:.4f})")
            print(f"Val Loss: {val_loss:.4f} (CE: {val_ce:.4f}, LB: {val_lb:.4f})")
            
            if self.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train/epoch_loss': train_loss,
                    'train/epoch_ce_loss': train_ce,
                    'train/epoch_lb_loss': train_lb,
                    'val/loss': val_loss,
                    'val/ce_loss': val_ce,
                    'val/lb_loss': val_lb,
                    'time/epoch_time': epoch_time
                })
            
            # Save checkpoint
            if epoch % save_every == 0:
                checkpoint_path = os.path.join(
                    self.checkpoint_dir,
                    f"checkpoint_epoch_{epoch}.pt"
                )
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    checkpoint_path
                )
                print(f"Checkpoint saved to {checkpoint_path}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_path = os.path.join(self.checkpoint_dir, "best_model.pt")
                save_checkpoint(
                    self.model,
                    self.optimizer,
                    epoch,
                    val_loss,
                    best_path
                )
                print(f"Best model saved! Val loss: {val_loss:.4f}")
                
                # Push to HuggingFace Hub
                if push_to_hub:
                    self.push_to_huggingface_hub(epoch)
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {format_time(total_time)}")
        
        # Save expert usage history
        usage_path = os.path.join(self.log_dir, "expert_usage_history.json")
        with open(usage_path, 'w') as f:
            json.dump(self.expert_usage_history, f, indent=2)
        print(f"Expert usage history saved to {usage_path}")
    
    def push_to_huggingface_hub(self, epoch: int):
        """Push model to HuggingFace Hub"""
        try:
            from huggingface_hub import HfApi, create_repo
            
            repo_name = f"{self.config.get('hf_username', 'user')}/moe-xsum-{self.config['routing']}"
            
            print(f"Pushing to HuggingFace Hub: {repo_name}")
            
            # Create repo if it doesn't exist
            try:
                create_repo(repo_name, exist_ok=True)
            except Exception as e:
                print(f"Repo creation warning: {e}")
            
            # Save model
            model_path = os.path.join(self.checkpoint_dir, f"hf_model_epoch_{epoch}")
            ensure_dir(model_path)
            
            torch.save(self.model.state_dict(), os.path.join(model_path, "pytorch_model.bin"))
            
            # Save config
            model_config = {
                'vocab_size': self.model.output_proj.out_features,
                'd_model': self.model.d_model,
                'routing': self.config['routing'],
                'num_experts': self.config.get('num_experts', config.NUM_EXPERTS),
                'top_k': self.config.get('top_k', config.TOP_K),
                'epoch': epoch
            }
            
            with open(os.path.join(model_path, "config.json"), 'w') as f:
                json.dump(model_config, f, indent=2)
            
            # Upload
            api = HfApi()
            api.upload_folder(
                folder_path=model_path,
                repo_id=repo_name,
                repo_type="model"
            )
            
            print(f"Successfully pushed to {repo_name}")
            
        except Exception as e:
            print(f"Failed to push to hub: {e}")


def main():
    # Load configuration
    config = load_config()
    
    # Debug mode configuration
    debug_config = config.get('debug', {})
    if debug_config.get('enabled', False):
        config['training']['train_samples'] = debug_config.get('train_samples', 100)
        config['training']['val_samples'] = debug_config.get('val_samples', 50)
        config['training']['num_epochs'] = debug_config.get('num_epochs', 2)
        print("DEBUG MODE: Using small dataset")
    
    # Set seed for reproducibility
    set_seed(config["training"]["seed"])
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Debug mode was already handled in debug_config section above
    
    # Initialize wandb
    use_wandb = config["wandb"]["enabled"]
    if use_wandb:
        wandb.init(
            project=config["wandb"]["project"],
            name=f"moe_{config['model']['routing']}_k{config['moe']['top_k']}_e{config['moe']['num_experts']}",
            config=config
        )
    
    # Load tokenizer
    print(f"Loading tokenizer: {config['model']['tokenizer']}")
    tokenizer = AutoTokenizer.from_pretrained(config['model']['tokenizer'])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    vocab_size = len(tokenizer)
    print(f"Vocabulary size: {vocab_size}")
    
    # Create data module
    print("Loading dataset...")
    data_module = XSumDataModule(
        tokenizer_name=config['model']['tokenizer'],
        batch_size=config['training']['batch_size'],
        max_source_length=config['dataset']['max_source_length'],
        max_target_length=config['dataset']['max_target_length'],
        num_workers=config['dataset']['num_workers'],
        train_samples=config['dataset']['train_samples'],
        val_samples=config['dataset']['val_samples']
    )
    
    train_loader = data_module.train_dataloader()
    val_loader = data_module.val_dataloader()
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches: {len(val_loader)}")
    
    # Create model
    print(f"\nCreating MoE Transformer with {config['model']['routing']} routing...")
    model = MoETransformer(
        vocab_size=vocab_size,
        d_model=config['model']['d_model'],
        n_heads=config['model']['n_heads'],
        n_layers=config['model']['n_layers'],
        d_ff=config['model']['d_ff'],
        num_experts=config['moe']['num_experts'],
        top_k=config['moe']['top_k'],
        router_type=config['model']['routing'],
        load_balancer_weight=config['moe']['load_balancer_weight'],
        use_load_balancer_loss=config['moe']['use_load_balancer'],
        dropout=config['model']['dropout'],
        pad_token_id=tokenizer.pad_token_id,
        use_moe_encoder=True,
        use_moe_decoder=True
    )
    
    total_params, trainable_params = count_parameters(model)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Create optimizer
    optimizer = AdamW(model.parameters(), lr=config['training']['learning_rate'])
    
    # Create scheduler
    total_steps = len(train_loader) * config['training']['epochs'] // config['training']['gradient_accumulation_steps']
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=config['training']['warmup_steps'],
        num_training_steps=total_steps
    )
    
    # Create checkpoint directory name
    checkpoint_dir = os.path.join(
        config['paths']['checkpoints'],
        f"moe_{config['model']['routing']}_k{config['moe']['top_k']}_e{config['moe']['num_experts']}"
    )
    
    # Create config dict for trainer
    config_dict = {
        'routing': config['model']['routing'],
        'num_experts': config['moe']['num_experts'],
        'top_k': config['moe']['top_k'],
        'use_load_balancer': config['moe']['use_load_balancer'],
        'load_balancer_weight': config['moe']['load_balancer_weight'],
        'gradient_accumulation_steps': config['training']['gradient_accumulation_steps'],
        'max_grad_norm': config['training']['max_grad_norm'],
        'hf_username': config['huggingface']['username']
    }
    
    # Create trainer
    trainer = MoETrainer(
        model=model,
        tokenizer=tokenizer,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config_dict=config_dict,
        checkpoint_dir=checkpoint_dir,
        log_dir=config['paths']['logs'],
        use_wandb=use_wandb
    )
    
    # Train
    trainer.train(
        num_epochs=config['training']['epochs'],
        save_every=1,
        push_to_hub=config['huggingface']['push_to_hub']
    )
    
    if use_wandb:
        wandb.finish()
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()