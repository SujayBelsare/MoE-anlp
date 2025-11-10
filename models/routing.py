"""
Routing Algorithms for Mixture of Experts
"""

import torch
import torch.nn.functional as F
from abc import ABC, abstractmethod
from typing import Tuple


class BaseRouting(ABC):
    """Abstract base class for routing algorithms"""
    
    @abstractmethod
    def route(
        self, 
        gate_logits: torch.Tensor, 
        top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Route tokens to experts based on gate logits.
        
        Args:
            gate_logits: Gate logits of shape (num_tokens, num_experts)
            top_k: Number of experts to route each token to
            
        Returns:
            expert_mask: Binary mask of shape (num_tokens, num_experts)
            expert_weights: Weights for each expert (num_tokens, num_experts)
            expert_indices: Indices of selected experts (num_tokens, top_k)
        """
        pass


class HashRouting(BaseRouting):
    """
    Hash-based routing: Assigns tokens to experts using a hash function.
    This is deterministic and doesn't use the gating network's output.
    """
    
    def __init__(self, num_experts: int):
        self.num_experts = num_experts
        
    def route(
        self, 
        gate_logits: torch.Tensor, 
        top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Hash-based routing. Each token is assigned to top_k experts based on its position.
        
        Args:
            gate_logits: Gate logits of shape (num_tokens, num_experts)
            top_k: Number of experts to route each token to
            
        Returns:
            expert_mask: Binary mask of shape (num_tokens, num_experts)
            expert_weights: Uniform weights (num_tokens, num_experts)
            expert_indices: Indices of selected experts (num_tokens, top_k)
        """
        num_tokens, num_experts = gate_logits.shape
        device = gate_logits.device
        
        # Create hash-based assignment
        # Use token position to determine expert assignment
        token_ids = torch.arange(num_tokens, device=device)
        
        # Hash each token to top_k experts
        expert_indices = torch.zeros(num_tokens, top_k, dtype=torch.long, device=device)
        for k in range(top_k):
            # Use a simple hash function: (position * prime + k) % num_experts
            expert_indices[:, k] = (token_ids * 7919 + k * 104729) % num_experts
        
        # Create expert mask
        expert_mask = torch.zeros(num_tokens, num_experts, device=device)
        for k in range(top_k):
            expert_mask[torch.arange(num_tokens), expert_indices[:, k]] = 1.0
        
        # Uniform weights for selected experts
        expert_weights = expert_mask / top_k
        
        return expert_mask, expert_weights, expert_indices


class TokenChoiceTopKRouting(BaseRouting):
    """
    Token-choice Top-K routing: Each token selects its top-k experts
    based on the gating network's output.
    """
    
    def __init__(self, use_softmax: bool = True):
        """
        Args:
            use_softmax: Whether to apply softmax to gate logits before top-k selection
        """
        self.use_softmax = use_softmax
        
    def route(
        self, 
        gate_logits: torch.Tensor, 
        top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Top-K routing based on gating network output.
        
        Args:
            gate_logits: Gate logits of shape (num_tokens, num_experts)
            top_k: Number of experts to route each token to
            
        Returns:
            expert_mask: Binary mask of shape (num_tokens, num_experts)
            expert_weights: Normalized weights (num_tokens, num_experts)
            expert_indices: Indices of selected experts (num_tokens, top_k)
        """
        num_tokens, num_experts = gate_logits.shape
        device = gate_logits.device
        
        # Apply softmax to get probabilities if specified
        if self.use_softmax:
            gate_probs = F.softmax(gate_logits, dim=-1)
        else:
            gate_probs = gate_logits
        
        # Get top-k experts for each token
        top_k_values, top_k_indices = torch.topk(
            gate_probs, 
            k=min(top_k, num_experts), 
            dim=-1
        )
        
        # Normalize the top-k weights to sum to 1
        top_k_weights = top_k_values / (top_k_values.sum(dim=-1, keepdim=True) + 1e-10)
        
        # Create expert mask and weights
        expert_mask = torch.zeros(num_tokens, num_experts, device=device)
        expert_weights = torch.zeros(num_tokens, num_experts, device=device)
        
        # Scatter the weights and create mask
        for k in range(min(top_k, num_experts)):
            indices = top_k_indices[:, k]
            expert_mask[torch.arange(num_tokens), indices] = 1.0
            expert_weights[torch.arange(num_tokens), indices] = top_k_weights[:, k]
        
        return expert_mask, expert_weights, top_k_indices


class ExpertChoiceRouting(BaseRouting):
    """
    Expert-choice routing: Each expert selects its top tokens
    (Alternative routing strategy, not required but useful)
    """
    
    def __init__(self, capacity_factor: float = 1.25):
        """
        Args:
            capacity_factor: Multiplier for expert capacity
        """
        self.capacity_factor = capacity_factor
        
    def route(
        self, 
        gate_logits: torch.Tensor, 
        top_k: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Expert-choice routing where experts select tokens.
        
        Args:
            gate_logits: Gate logits of shape (num_tokens, num_experts)
            top_k: Number of tokens each expert should process
            
        Returns:
            expert_mask: Binary mask of shape (num_tokens, num_experts)
            expert_weights: Weights for each expert (num_tokens, num_experts)
            expert_indices: Indices of selected experts (num_tokens, top_k)
        """
        num_tokens, num_experts = gate_logits.shape
        device = gate_logits.device
        
        # Compute expert capacity
        capacity = int(self.capacity_factor * num_tokens / num_experts)
        
        # Apply softmax
        gate_probs = F.softmax(gate_logits, dim=-1)
        
        # Transpose to get (num_experts, num_tokens)
        gate_probs_transposed = gate_probs.t()
        
        # Each expert selects top tokens
        expert_mask = torch.zeros(num_tokens, num_experts, device=device)
        expert_weights = torch.zeros(num_tokens, num_experts, device=device)
        
        for expert_idx in range(num_experts):
            # Get top tokens for this expert
            top_values, top_indices = torch.topk(
                gate_probs_transposed[expert_idx],
                k=min(capacity, num_tokens)
            )
            
            # Set mask and weights
            expert_mask[top_indices, expert_idx] = 1.0
            expert_weights[top_indices, expert_idx] = top_values
        
        # Normalize weights per token
        token_sums = expert_weights.sum(dim=-1, keepdim=True)
        expert_weights = expert_weights / (token_sums + 1e-10)
        
        # Get expert indices for each token
        expert_indices = torch.topk(expert_mask, k=top_k, dim=-1)[1]
        
        return expert_mask, expert_weights, expert_indices