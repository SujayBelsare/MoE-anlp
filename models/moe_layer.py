import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class Expert(nn.Module):
    """Simple Feed-Forward Network Expert"""
    
    def __init__(self, d_model: int, expert_hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, expert_hidden_dim)
        self.fc2 = nn.Linear(expert_hidden_dim, d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class GatingNetwork(nn.Module):
    """Gating network for routing tokens to experts"""
    
    def __init__(self, d_model: int, num_experts: int):
        super().__init__()
        self.gate = nn.Linear(d_model, num_experts, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Gating scores of shape (batch_size, seq_len, num_experts)
        """
        return self.gate(x)


class SparseMoELayer(nn.Module):
    """Sparse Mixture of Experts Layer"""
    
    def __init__(
        self,
        d_model: int,
        num_experts: int,
        expert_hidden_dim: int,
        top_k: int,
        router_type: str = "top_k",
        dropout: float = 0.1,
        use_load_balancer: bool = False,
    ):
        super().__init__()
        self.d_model = d_model
        self.num_experts = num_experts
        self.top_k = top_k
        self.router_type = router_type
        self.use_load_balancer = use_load_balancer
        
        # Create experts
        self.experts = nn.ModuleList([
            Expert(d_model, expert_hidden_dim, dropout)
            for _ in range(num_experts)
        ])
        
        # Gating network
        self.gating_network = GatingNetwork(d_model, num_experts)
        
        # For tracking expert usage
        self.register_buffer("expert_usage", torch.zeros(num_experts))
        self.register_buffer("total_tokens", torch.tensor(0))
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            output: Output tensor of shape (batch_size, seq_len, d_model)
            aux_loss: Load balancing auxiliary loss (if enabled)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Flatten for easier processing
        x_flat = x.view(-1, d_model)  # (batch_size * seq_len, d_model)
        
        # Get gating scores
        gate_logits = self.gating_network(x_flat)  # (batch_size * seq_len, num_experts)
        
        # Apply routing
        if self.router_type == "hash":
            expert_outputs, expert_weights, expert_indices = self._hash_routing(x_flat, gate_logits)
        else:  # top_k routing
            expert_outputs, expert_weights, expert_indices = self._top_k_routing(x_flat, gate_logits)
        
        # Update expert usage statistics
        if self.training:
            self._update_expert_usage(expert_indices)
        
        # Compute load balancing loss
        aux_loss = None
        if self.use_load_balancer and self.training:
            aux_loss = self._compute_load_balance_loss(gate_logits, expert_indices)
        
        # Reshape output
        output = expert_outputs.view(batch_size, seq_len, d_model)
        
        return output, aux_loss
    
    def _top_k_routing(
        self, x: torch.Tensor, gate_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Token-choice top-k routing"""
        num_tokens = x.shape[0]
        
        # Get top-k experts for each token
        gate_scores = F.softmax(gate_logits, dim=-1)
        top_k_scores, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        
        # Normalize the top-k scores
        top_k_scores = top_k_scores / top_k_scores.sum(dim=-1, keepdim=True)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each token through its selected experts
        for i in range(num_tokens):
            token_output = torch.zeros_like(x[i])
            for j in range(self.top_k):
                expert_idx = top_k_indices[i, j].item()
                expert_weight = top_k_scores[i, j]
                expert_out = self.experts[expert_idx](x[i:i+1])
                token_output += expert_weight * expert_out.squeeze(0)
            output[i] = token_output
        
        return output, top_k_scores, top_k_indices
    
    def _hash_routing(
        self, x: torch.Tensor, gate_logits: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Hash-based routing"""
        num_tokens = x.shape[0]
        
        # Use hash function based on token position
        expert_indices = torch.arange(num_tokens, device=x.device) % self.num_experts
        expert_indices = expert_indices.unsqueeze(1)  # (num_tokens, 1)
        
        # Create uniform weights
        expert_weights = torch.ones((num_tokens, 1), device=x.device)
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Process each token through its assigned expert
        for i in range(num_tokens):
            expert_idx = expert_indices[i, 0].item()
            output[i] = self.experts[expert_idx](x[i:i+1]).squeeze(0)
        
        return output, expert_weights, expert_indices
    
    def _update_expert_usage(self, expert_indices: torch.Tensor):
        """Update expert usage statistics"""
        with torch.no_grad():
            for idx in expert_indices.flatten():
                self.expert_usage[idx] += 1
            self.total_tokens += expert_indices.numel()
    
    def _compute_load_balance_loss(
        self, gate_logits: torch.Tensor, expert_indices: torch.Tensor
    ) -> torch.Tensor:
        """Compute load balancing auxiliary loss"""
        # Calculate the fraction of tokens routed to each expert
        gate_probs = F.softmax(gate_logits, dim=-1)  # (num_tokens, num_experts)
        
        # Mean probability of routing to each expert
        mean_probs = gate_probs.mean(dim=0)  # (num_experts,)
        
        # Fraction of tokens actually assigned to each expert
        num_tokens = expert_indices.shape[0]
        expert_counts = torch.zeros(self.num_experts, device=expert_indices.device)
        for idx in expert_indices.flatten():
            expert_counts[idx] += 1
        expert_fractions = expert_counts / num_tokens
        
        # Load balance loss: encourage uniform distribution
        # Loss = num_experts * sum(mean_probs * expert_fractions)
        loss = self.num_experts * (mean_probs * expert_fractions).sum()
        
        return loss
    
    def get_expert_usage_stats(self):
        """Get expert usage statistics"""
        if self.total_tokens == 0:
            return torch.zeros(self.num_experts)
        return self.expert_usage / self.total_tokens
    
    def reset_expert_usage_stats(self):
        """Reset expert usage statistics"""
        self.expert_usage.zero_()
        self.total_tokens.zero_()