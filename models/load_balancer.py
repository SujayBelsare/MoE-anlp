"""
Load Balancer for Mixture of Experts
Implements auxiliary loss to encourage balanced expert usage
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoadBalancer:
    """
    Load Balancer that computes an auxiliary loss to encourage
    balanced distribution of tokens across experts.
    
    Implements the load balancing loss from Switch Transformers paper:
    https://arxiv.org/abs/2101.03961
    """
    
    def __init__(self, num_experts: int, loss_weight: float = 0.01):
        """
        Args:
            num_experts: Total number of experts
            loss_weight: Weight for the load balancing loss
        """
        self.num_experts = num_experts
        self.loss_weight = loss_weight
        
    def get_loss(
        self,
        gate_logits: torch.Tensor,
        expert_mask: torch.Tensor,
        expert_indices: torch.Tensor = None
    ) -> torch.Tensor:
        """
        Compute the load balancing loss.
        
        The loss encourages uniform distribution of tokens across experts by
        penalizing the product of:
        1. Fraction of router probability allocated to each expert
        2. Fraction of tokens dispatched to each expert
        
        Args:
            gate_logits: Gate logits of shape (num_tokens, num_experts)
            expert_mask: Binary mask of shape (num_tokens, num_experts)
            expert_indices: Optional indices of selected experts (num_tokens, top_k)
            
        Returns:
            load_balancer_loss: Scalar tensor
        """
        num_tokens, num_experts = gate_logits.shape
        
        # Compute router probabilities
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Compute fraction of router probability allocated to each expert
        # This is the mean probability assigned to each expert across all tokens
        prob_per_expert = gate_probs.mean(dim=0)  # (num_experts,)
        
        # Compute fraction of tokens dispatched to each expert
        # This is the fraction of tokens that selected this expert
        tokens_per_expert = expert_mask.sum(dim=0) / num_tokens  # (num_experts,)
        
        # Load balancing loss: num_experts * sum(prob_per_expert * tokens_per_expert)
        # We want both distributions to be uniform (1/num_experts for each expert)
        # When both are uniform, this loss equals 1. When imbalanced, it's higher.
        load_balancer_loss = num_experts * torch.sum(
            prob_per_expert * tokens_per_expert
        )
        
        return self.loss_weight * load_balancer_loss
    
    def get_importance_loss(
        self,
        gate_logits: torch.Tensor,
        expert_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Alternative formulation: Importance loss
        Minimizes variance in expert usage.
        
        Args:
            gate_logits: Gate logits of shape (num_tokens, num_experts)
            expert_mask: Binary mask of shape (num_tokens, num_experts)
            
        Returns:
            importance_loss: Scalar tensor
        """
        # Compute how many tokens each expert receives
        tokens_per_expert = expert_mask.sum(dim=0)  # (num_experts,)
        
        # Compute the coefficient of variation (std/mean)
        # Lower CV means more balanced distribution
        mean_tokens = tokens_per_expert.mean()
        std_tokens = tokens_per_expert.std()
        
        # Avoid division by zero
        cv = std_tokens / (mean_tokens + 1e-10)
        
        return self.loss_weight * cv
    
    def get_z_loss(
        self,
        gate_logits: torch.Tensor,
        z_loss_weight: float = 1e-3
    ) -> torch.Tensor:
        """
        Z-loss: Encourages router logits to remain small.
        Helps with training stability.
        
        Args:
            gate_logits: Gate logits of shape (num_tokens, num_experts)
            z_loss_weight: Weight for z-loss
            
        Returns:
            z_loss: Scalar tensor
        """
        # Compute log(sum(exp(logits)))^2 for each token
        log_z = torch.logsumexp(gate_logits, dim=-1)
        z_loss = torch.mean(log_z ** 2)
        
        return z_loss_weight * z_loss
    
    def get_combined_loss(
        self,
        gate_logits: torch.Tensor,
        expert_mask: torch.Tensor,
        expert_indices: torch.Tensor = None,
        use_importance_loss: bool = False,
        use_z_loss: bool = True
    ) -> torch.Tensor:
        """
        Compute combined load balancing loss with optional components.
        
        Args:
            gate_logits: Gate logits of shape (num_tokens, num_experts)
            expert_mask: Binary mask of shape (num_tokens, num_experts)
            expert_indices: Optional indices of selected experts
            use_importance_loss: Whether to include importance loss
            use_z_loss: Whether to include z-loss
            
        Returns:
            combined_loss: Scalar tensor
        """
        # Primary load balancing loss
        lb_loss = self.get_loss(gate_logits, expert_mask, expert_indices)
        
        # Optional importance loss
        if use_importance_loss:
            imp_loss = self.get_importance_loss(gate_logits, expert_mask)
            lb_loss = lb_loss + imp_loss
        
        # Optional z-loss for stability
        if use_z_loss:
            z_loss = self.get_z_loss(gate_logits)
            lb_loss = lb_loss + z_loss
        
        return lb_loss


class CapacityBalancer:
    """
    Implements capacity-based load balancing where each expert
    has a maximum capacity of tokens it can process.
    """
    
    def __init__(
        self,
        num_experts: int,
        capacity_factor: float = 1.25,
        drop_tokens: bool = True
    ):
        """
        Args:
            num_experts: Total number of experts
            capacity_factor: Multiplier for expert capacity
            drop_tokens: Whether to drop tokens that exceed capacity
        """
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.drop_tokens = drop_tokens
        
    def apply_capacity(
        self,
        expert_mask: torch.Tensor,
        expert_weights: torch.Tensor,
        num_tokens: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Apply capacity constraints to expert assignments.
        
        Args:
            expert_mask: Binary mask of shape (num_tokens, num_experts)
            expert_weights: Weights for each expert (num_tokens, num_experts)
            num_tokens: Total number of tokens
            
        Returns:
            capacity_mask: Updated mask respecting capacity
            updated_weights: Updated weights
            overflow_mask: Mask of tokens that were dropped
        """
        # Compute expert capacity
        capacity = int(self.capacity_factor * num_tokens / self.num_experts)
        
        # Track how many tokens each expert has received
        expert_counts = torch.zeros(self.num_experts, device=expert_mask.device)
        
        # Create capacity-constrained mask
        capacity_mask = torch.zeros_like(expert_mask)
        overflow_mask = torch.zeros(num_tokens, dtype=torch.bool, device=expert_mask.device)
        
        # Process tokens in order (you might want to randomize this)
        for token_idx in range(num_tokens):
            for expert_idx in range(self.num_experts):
                if expert_mask[token_idx, expert_idx] > 0:
                    if expert_counts[expert_idx] < capacity:
                        capacity_mask[token_idx, expert_idx] = 1.0
                        expert_counts[expert_idx] += 1
                    elif self.drop_tokens:
                        overflow_mask[token_idx] = True
        
        # Update weights based on capacity mask
        updated_weights = expert_weights * capacity_mask
        
        # Renormalize weights per token
        token_sums = updated_weights.sum(dim=-1, keepdim=True)
        updated_weights = updated_weights / (token_sums + 1e-10)
        
        return capacity_mask, updated_weights, overflow_mask