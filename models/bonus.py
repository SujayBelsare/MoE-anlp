"""
Bonus implementations for ANLP Assignment 3
- Bonus 2: Grouped Query Attention (GQA)
- Bonus 3: LoRA-based Experts
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class GroupedQueryAttention(nn.Module):
    """
    Bonus 2: Grouped Query Attention implementation from scratch
    
    GQA reduces the number of key-value heads while keeping multiple query heads,
    providing a middle ground between Multi-Head Attention and Multi-Query Attention.
    """
    
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension
            num_heads: Number of query heads
            num_kv_heads: Number of key-value heads (< num_heads)
            dropout: Dropout probability
        """
        super().__init__()
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads"
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = d_model // num_heads
        self.num_queries_per_kv = num_heads // num_kv_heads
        
        # Query projection (full number of heads)
        self.q_proj = nn.Linear(d_model, num_heads * self.head_dim, bias=False)
        
        # Key and Value projections (reduced number of heads)
        self.k_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, num_kv_heads * self.head_dim, bias=False)
        
        # Output projection
        self.o_proj = nn.Linear(num_heads * self.head_dim, d_model, bias=False)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = self.head_dim ** -0.5
    
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            query: (batch_size, seq_len_q, d_model)
            key: (batch_size, seq_len_k, d_model)
            value: (batch_size, seq_len_v, d_model)
            attn_mask: (seq_len_q, seq_len_k) or (batch_size, seq_len_q, seq_len_k)
            key_padding_mask: (batch_size, seq_len_k)
        
        Returns:
            output: (batch_size, seq_len_q, d_model)
            attention_weights: (batch_size, num_heads, seq_len_q, seq_len_k)
        """
        batch_size, seq_len_q, _ = query.shape
        seq_len_k = key.shape[1]
        
        # Project queries, keys, values
        # Q: (batch_size, seq_len_q, num_heads, head_dim)
        Q = self.q_proj(query).view(batch_size, seq_len_q, self.num_heads, self.head_dim)
        
        # K, V: (batch_size, seq_len_k, num_kv_heads, head_dim)
        K = self.k_proj(key).view(batch_size, seq_len_k, self.num_kv_heads, self.head_dim)
        V = self.v_proj(value).view(batch_size, seq_len_k, self.num_kv_heads, self.head_dim)
        
        # Transpose for attention computation
        # Q: (batch_size, num_heads, seq_len_q, head_dim)
        Q = Q.transpose(1, 2)
        
        # K, V: (batch_size, num_kv_heads, seq_len_k, head_dim)
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Expand K and V to match number of query heads
        # Repeat each KV head for multiple query heads
        # (batch_size, num_kv_heads, seq_len_k, head_dim) -> 
        # (batch_size, num_heads, seq_len_k, head_dim)
        K = K.repeat_interleave(self.num_queries_per_kv, dim=1)
        V = V.repeat_interleave(self.num_queries_per_kv, dim=1)
        
        # Compute attention scores
        # (batch_size, num_heads, seq_len_q, seq_len_k)
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # Apply attention mask
        if attn_mask is not None:
            if attn_mask.dim() == 2:
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)
            elif attn_mask.dim() == 3:
                attn_mask = attn_mask.unsqueeze(1)
            attn_scores = attn_scores + attn_mask
        
        # Apply key padding mask
        if key_padding_mask is not None:
            # (batch_size, seq_len_k) -> (batch_size, 1, 1, seq_len_k)
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(2)
            attn_scores = attn_scores.masked_fill(key_padding_mask, float('-inf'))
        
        # Compute attention weights
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        # (batch_size, num_heads, seq_len_q, head_dim)
        output = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        # (batch_size, seq_len_q, num_heads * head_dim)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len_q, -1)
        output = self.o_proj(output)
        
        return output, attn_weights


class LoRALinear(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer from scratch
    
    Instead of fine-tuning W, we freeze W and learn:
    ΔW = B @ A, where A ∈ R^{r×d_in}, B ∈ R^{d_out×r}, r << min(d_in, d_out)
    
    Forward: y = Wx + (B @ A)x = Wx + BAx
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        """
        Args:
            in_features: Input dimension
            out_features: Output dimension
            rank: LoRA rank (r)
            alpha: LoRA scaling parameter
            dropout: Dropout probability
        """
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # Original weight (frozen)
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        self.weight.requires_grad = False
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """Initialize parameters"""
        # Initialize original weight
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        
        # Initialize LoRA matrices
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (*, in_features)
        Returns:
            output: (*, out_features)
        """
        # Original forward pass
        result = F.linear(x, self.weight)
        
        # LoRA forward pass: x @ A^T @ B^T
        x_dropped = self.dropout(x)
        lora_out = F.linear(F.linear(x_dropped, self.lora_A), self.lora_B)
        
        return result + lora_out * self.scaling


class LoRAExpert(nn.Module):
    """
    Bonus 3: Expert module using LoRA instead of full linear layers
    
    This reduces the number of trainable parameters per expert significantly
    while maintaining expressiveness.
    """
    
    def __init__(
        self,
        d_model: int,
        expert_hidden_dim: int,
        lora_rank: int = 8,
        lora_alpha: int = 16,
        dropout: float = 0.1,
    ):
        """
        Args:
            d_model: Model dimension
            expert_hidden_dim: Hidden dimension of expert FFN
            lora_rank: LoRA rank
            lora_alpha: LoRA alpha parameter
            dropout: Dropout probability
        """
        super().__init__()
        
        # Use LoRA layers instead of regular linear layers
        self.fc1 = LoRALinear(
            d_model,
            expert_hidden_dim,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=dropout
        )
        
        self.fc2 = LoRALinear(
            expert_hidden_dim,
            d_model,
            rank=lora_rank,
            alpha=lora_alpha,
            dropout=dropout
        )
        
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, seq_len, d_model)
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    def count_parameters(self) -> dict:
        """Count trainable vs frozen parameters"""
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        frozen = sum(p.numel() for p in self.parameters() if not p.requires_grad)
        return {
            'trainable': trainable,
            'frozen': frozen,
            'total': trainable + frozen,
            'compression_ratio': frozen / trainable if trainable > 0 else 0
        }


# Example usage and testing
if __name__ == "__main__":
    print("Testing Bonus Implementations...\n")
    
    # Test GQA
    print("=" * 50)
    print("Testing Grouped Query Attention")
    print("=" * 50)
    
    gqa = GroupedQueryAttention(
        d_model=512,
        num_heads=8,
        num_kv_heads=4,
        dropout=0.1
    )
    
    batch_size, seq_len, d_model = 2, 10, 512
    query = torch.randn(batch_size, seq_len, d_model)
    key = torch.randn(batch_size, seq_len, d_model)
    value = torch.randn(batch_size, seq_len, d_model)
    
    output, attn_weights = gqa(query, key, value)
    print(f"Input shape: {query.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Attention weights shape: {attn_weights.shape}")
    print(f"✓ GQA test passed!\n")
    
    # Test LoRA Expert
    print("=" * 50)
    print("Testing LoRA-based Expert")
    print("=" * 50)
    
    lora_expert = LoRAExpert(
        d_model=512,
        expert_hidden_dim=2048,
        lora_rank=8,
        lora_alpha=16,
        dropout=0.1
    )
    
    x = torch.randn(batch_size, seq_len, d_model)
    output = lora_expert(x)
    
    params = lora_expert.count_parameters()
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {output.shape}")
    print(f"\nParameter counts:")
    print(f"  Trainable: {params['trainable']:,}")
    print(f"  Frozen: {params['frozen']:,}")
    print(f"  Total: {params['total']:,}")
    print(f"  Compression ratio: {params['compression_ratio']:.2f}x")
    print(f"✓ LoRA Expert test passed!\n")