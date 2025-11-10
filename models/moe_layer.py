"""
Sparse Mixture of Experts Layer Implementation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Expert(nn.Module):
    """
    A simple Feed-Forward Network that acts as an expert.
    Architecture: Linear -> Activation -> Dropout -> Linear
    """
    
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff)
        self.w2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        return self.w2(self.dropout(self.activation(self.w1(x))))


class GatingNetwork(nn.Module):
    """
    Gating network that produces routing probabilities for each token.
    Uses a simple linear layer followed by softmax.
    """
    
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Gate logits of shape (batch_size, seq_len, num_experts)
        """
        return self.gate(x)


class SparseMoELayer(nn.Module):
    """
    Sparse Mixture of Experts Layer that replaces a standard FFN.
    """
    
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        router,
        load_balancer=None,
        dropout: float = 0.1,
        use_load_balancer_loss: bool = False
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.use_load_balancer_loss = use_load_balancer_loss
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, d_ff, dropout) for _ in range(num_experts)
        ])
        
        # Gating network
        self.gating_net = GatingNetwork(d_model, num_experts)
        
        # Router
        self.router = router
        
        # Load balancer (optional)
        self.load_balancer = load_balancer
        
        # For tracking expert usage
        self.register_buffer("expert_usage", torch.zeros(num_experts))

        self.expert_usage = torch.zeros(num_experts)
        
    def forward(
        self, 
        x: torch.Tensor,
        return_load_balancer_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the MoE layer.
        
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            return_load_balancer_loss: Whether to return load balancer loss
            
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            load_balancer_loss: Optional load balancing loss (if enabled)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Flatten batch and sequence dimensions for easier processing
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)
        
        # Get gate logits
        gate_logits = self.gating_net(x_flat)  # (batch_size * seq_len, num_experts)
        
        # Route tokens to experts
        expert_mask, expert_weights, expert_indices = self.router.route(
            gate_logits, self.top_k
        )
        
        # Initialize output
        output = torch.zeros_like(x_flat)
        
        # Process each expert
        for expert_idx in range(self.num_experts):
            # Get mask for tokens assigned to this expert
            expert_token_mask = expert_mask[:, expert_idx]  # (batch_size * seq_len,)
            
            if expert_token_mask.any():
                # Get tokens for this expert
                expert_input = x_flat[expert_token_mask]
                
                # Process through expert
                expert_output = self.experts[expert_idx](expert_input)
                
                # Get weights for this expert
                weights = expert_weights[expert_token_mask, expert_idx].unsqueeze(-1)
                
                # Add weighted expert output to final output
                output[expert_token_mask] += weights * expert_output
                
                # Update expert usage tracking
                self.expert_usage[expert_idx] += expert_token_mask.sum().item()
        
        # Reshape output back to original shape
        output = output.view(batch_size, seq_len, d_model)
        
        # Compute load balancer loss if needed
        load_balancer_loss = None
        if return_load_balancer_loss and self.use_load_balancer_loss and self.load_balancer:
            load_balancer_loss = self.load_balancer.get_loss(
                gate_logits, expert_mask, expert_indices
            )
        
        return output, load_balancer_loss
    
    def get_expert_usage(self):
        """Get the current expert usage statistics"""
        return self.expert_usage.cpu().numpy()
    
    def reset_expert_usage(self):
        """Reset expert usage tracking"""
        self.expert_usage.zero_()