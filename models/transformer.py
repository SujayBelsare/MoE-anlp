"""
Transformer architecture with Mixture of Experts layers
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from models.moe_layer import SparseMoELayer
from configs import load_config

# Load default configuration
config = load_config()


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))
        self.pe: torch.Tensor
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (batch_size, seq_len, d_model)
        Returns:
            Tensor with positional encoding added
        """
        return x + self.pe[:, :x.size(1), :]


class MultiHeadAttention(nn.Module):
    """Multi-head attention mechanism"""
    
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Linear projections
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.out_linear = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: (batch_size, seq_len, d_model)
            key: (batch_size, seq_len, d_model)
            value: (batch_size, seq_len, d_model)
            mask: Optional attention mask
        Returns:
            Output tensor of shape (batch_size, seq_len, d_model)
        """
        batch_size = query.size(0)
        
        # Linear projections and reshape for multi-head
        Q = self.q_linear(query).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.k_linear(key).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(value).view(batch_size, -1, self.n_heads, self.d_k).transpose(1, 2)
        
        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attention_weights = torch.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        
        # Apply attention to values
        context = torch.matmul(attention_weights, V)
        
        # Concatenate heads
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
        
        # Final linear projection
        output = self.out_linear(context)
        
        return output


class TransformerEncoderLayer(nn.Module):
    """Single Transformer encoder layer with MoE"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        router,
        load_balancer=None,
        use_load_balancer_loss: bool = False,
        dropout: float = 0.1,
        use_moe: bool = True
    ):
        super().__init__()
        
        self.use_moe = use_moe
        self.use_load_balancer_loss = use_load_balancer_loss
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward or MoE
        if use_moe:
            self.ffn = SparseMoELayer(
                d_model, d_ff, num_experts, top_k, router, load_balancer, dropout,
                use_load_balancer_loss
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_load_balancer_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
            mask: Optional attention mask
            return_load_balancer_loss: Whether to return load balancer loss
        Returns:
            output: Output tensor
            load_balancer_loss: Optional load balancer loss
        """
        # Self-attention with residual connection
        attn_output = self.self_attn(x, x, x, mask)
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed-forward or MoE with residual connection
        if self.use_moe:
            ffn_output, lb_loss = self.ffn(x, return_load_balancer_loss)
        else:
            ffn_output = self.ffn(x)
            lb_loss = None
        
        x = self.norm2(x + self.dropout(ffn_output))
        
        return x, lb_loss


class TransformerDecoderLayer(nn.Module):
    """Single Transformer decoder layer with MoE"""
    
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        num_experts: int,
        top_k: int,
        router,
        load_balancer=None,
        use_load_balancer_loss: bool = False,
        dropout: float = 0.1,
        use_moe: bool = True
    ):
        super().__init__()
        
        self.use_moe = use_moe
        self.use_load_balancer_loss = use_load_balancer_loss
        
        # Self-attention
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Cross-attention
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        
        # Feed-forward or MoE
        if use_moe:
            self.ffn = SparseMoELayer(
                d_model, d_ff, num_experts, top_k, router, load_balancer, dropout,
                use_load_balancer_loss
            )
        else:
            self.ffn = nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(d_ff, d_model),
                nn.Dropout(dropout)
            )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        return_load_balancer_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            x: Input tensor of shape (batch_size, tgt_seq_len, d_model)
            encoder_output: Encoder output of shape (batch_size, src_seq_len, d_model)
            src_mask: Optional source mask
            tgt_mask: Optional target mask
            return_load_balancer_loss: Whether to return load balancer loss
        Returns:
            output: Output tensor
            load_balancer_loss: Optional load balancer loss
        """
        # Self-attention
        self_attn_output = self.self_attn(x, x, x, tgt_mask)
        x = self.norm1(x + self.dropout(self_attn_output))
        
        # Cross-attention
        cross_attn_output = self.cross_attn(x, encoder_output, encoder_output, src_mask)
        x = self.norm2(x + self.dropout(cross_attn_output))
        
        # Feed-forward or MoE
        if self.use_moe:
            ffn_output, lb_loss = self.ffn(x, return_load_balancer_loss)
        else:
            ffn_output = self.ffn(x)
            lb_loss = None
        
        x = self.norm3(x + self.dropout(ffn_output))
        
        return x, lb_loss


class MoETransformer(nn.Module):
    """
    Complete Transformer model with Mixture of Experts layers
    for sequence-to-sequence tasks (e.g., summarization)
    """
    
    def __init__(
        self,
        vocab_size: int,
        d_model: Optional[int] = None,
        n_heads: Optional[int] = None,
        n_layers: Optional[int] = None,
        d_ff: Optional[int] = None,
        num_experts: Optional[int] = None,
        top_k: Optional[int] = None,
        router_type: str = "topk",  # "topk" or "hash"
        load_balancer_weight: Optional[float] = None,
        use_load_balancer_loss: bool = False,
        dropout_rate: Optional[float] = None,
        max_len: int = 5000,
        pad_token_id: int = 0,
        use_moe_encoder: bool = True,
        use_moe_decoder: bool = True
    ):
        # Use config defaults if not provided
        self.d_model = d_model or config['model']['d_model']
        self.n_heads = n_heads or config['model']['n_heads']
        self.n_layers = n_layers or config['model']['n_layers']
        self.d_ff = d_ff or config['model']['d_ff']
        self.num_experts = num_experts or config['moe']['num_experts']
        self.top_k = top_k or config['moe']['top_k']
        self.load_balancer_weight = load_balancer_weight or config['moe']['load_balancer_weight']
        self.dropout_rate = dropout_rate or config['model']['dropout_rate']
        super().__init__()
        
        self.pad_token_id = pad_token_id
        self.use_load_balancer_loss = use_load_balancer_loss
        
        # Import routing algorithms
        from models.routing import HashRouting, TokenChoiceTopKRouting
        from models.load_balancer import LoadBalancer
        
        # Create router
        if router_type == "hash":
            router = HashRouting(self.num_experts)
        elif router_type == "topk":
            router = TokenChoiceTopKRouting(use_softmax=True)
        else:
            raise ValueError(f"Unknown router type: {router_type}")
        
        # Create load balancer
        load_balancer = LoadBalancer(self.num_experts, self.load_balancer_weight) if use_load_balancer_loss else None
        
        # Embeddings
        self.encoder_embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=self.pad_token_id)
        self.decoder_embedding = nn.Embedding(vocab_size, self.d_model, padding_idx=self.pad_token_id)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(self.d_model, max_len)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                self.d_model, self.n_heads, self.d_ff, self.num_experts, self.top_k,
                router, load_balancer, use_load_balancer_loss, self.dropout_rate, use_moe_encoder
            )
            for _ in range(self.n_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayer(
                self.d_model, self.n_heads, self.d_ff, self.num_experts, self.top_k,
                router, load_balancer, use_load_balancer_loss, self.dropout_rate, use_moe_decoder
            )
            for _ in range(self.n_layers)
        ])
        
        # Output projection
        self.output_proj = nn.Linear(self.d_model, vocab_size)
        
        self.dropout : torch.nn.Dropout = nn.Dropout(self.dropout_rate) 
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def create_masks(
        self,
        src: torch.Tensor,
        tgt: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Create attention masks for source and target sequences
        
        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
            tgt: Target tensor of shape (batch_size, tgt_seq_len)
        Returns:
            src_mask: Source padding mask
            tgt_mask: Target mask (padding + causal)
        """
        # Source padding mask
        src_mask = (src != self.pad_token_id).unsqueeze(1).unsqueeze(2)
        
        if tgt is not None:
            # Target padding mask
            tgt_pad_mask = (tgt != self.pad_token_id).unsqueeze(1).unsqueeze(2)
            
            # Causal mask
            tgt_seq_len = tgt.size(1)
            causal_mask = torch.tril(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device)).bool()
            causal_mask = causal_mask.unsqueeze(0).unsqueeze(0)
            
            # Combine masks
            tgt_mask = tgt_pad_mask & causal_mask
        else:
            tgt_mask = None
        
        return src_mask, tgt_mask
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: torch.Tensor,
        return_load_balancer_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Encode source sequence
        
        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
            src_mask: Source mask
            return_load_balancer_loss: Whether to return load balancer loss
        Returns:
            encoder_output: Encoded representation
            total_lb_loss: Total load balancer loss from all layers
        """
        # Embed and add positional encoding
        x = self.encoder_embedding(src) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply encoder layers
        total_lb_loss: Optional[torch.Tensor] = None
        for layer in self.encoder_layers:
            x, lb_loss = layer(x, src_mask, return_load_balancer_loss)
            if lb_loss is not None:
                total_lb_loss = lb_loss if total_lb_loss is None else total_lb_loss + lb_loss
        
        return x, total_lb_loss if return_load_balancer_loss else None
    
    def decode(
        self,
        tgt: torch.Tensor,
        encoder_output: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        return_load_balancer_loss: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Decode target sequence
        
        Args:
            tgt: Target tensor of shape (batch_size, tgt_seq_len)
            encoder_output: Encoder output
            src_mask: Source mask
            tgt_mask: Target mask
            return_load_balancer_loss: Whether to return load balancer loss
        Returns:
            decoder_output: Decoded representation
            total_lb_loss: Total load balancer loss from all layers
        """
        # Embed and add positional encoding
        x = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        x = self.pos_encoding(x)
        x = self.dropout(x)
        
        # Apply decoder layers
        total_lb_loss: Optional[torch.Tensor] = None
        for layer in self.decoder_layers:
            x, lb_loss = layer(x, encoder_output, src_mask, tgt_mask, return_load_balancer_loss)
            if lb_loss is not None:
                total_lb_loss = lb_loss if total_lb_loss is None else total_lb_loss + lb_loss
        
        return x, total_lb_loss if return_load_balancer_loss else None
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        return_load_balancer_loss: Optional[bool] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through the model
        
        Args:
            src: Source tensor of shape (batch_size, src_seq_len)
            tgt: Target tensor of shape (batch_size, tgt_seq_len)
            return_load_balancer_loss: Whether to return load balancer loss
        Returns:
            logits: Output logits of shape (batch_size, tgt_seq_len, vocab_size)
            total_lb_loss: Total load balancer loss
        """
        if return_load_balancer_loss is None:
            return_load_balancer_loss = self.use_load_balancer_loss
        
        # Create masks
        src_mask, tgt_mask = self.create_masks(src, tgt)
        
        # Encode
        encoder_output, enc_lb_loss = self.encode(src, src_mask, return_load_balancer_loss)
        
        # Decode
        decoder_output, dec_lb_loss = self.decode(
            tgt, encoder_output, src_mask, tgt_mask, return_load_balancer_loss
        )
        
        # Project to vocabulary
        logits = self.output_proj(decoder_output)
        
        # Combine load balancer losses
        total_lb_loss : Optional[torch.Tensor] = None
        if return_load_balancer_loss:
            if enc_lb_loss is not None:
                total_lb_loss = enc_lb_loss
            if dec_lb_loss is not None:
                total_lb_loss = dec_lb_loss if total_lb_loss is None else total_lb_loss + dec_lb_loss
        
        return logits, total_lb_loss
    
    def get_expert_usage(self):
        """Get expert usage statistics from all MoE layers"""
        usage_stats = {
            'encoder': [],
            'decoder': []
        }
        
        for i, layer in enumerate(self.encoder_layers):
            ffn = getattr(layer, "ffn", None)
            if ffn is not None and hasattr(ffn, "get_expert_usage"):
                method = getattr(ffn, "get_expert_usage", None)
                if callable(method):
                    usage_stats["encoder"].append(method())

        
        for i, layer in enumerate(self.decoder_layers):
            ffn = getattr(layer, "ffn", None)
            if ffn is not None and hasattr(ffn, "get_expert_usage"):
                method = getattr(ffn, "get_expert_usage", None)
                if callable(method):
                    usage_stats['decoder'].append(method())
        
        return usage_stats
    
    def reset_expert_usage(self):
        """Reset expert usage tracking"""
        for layer in self.encoder_layers:
            ffn = getattr(layer, "ffn", None)
            if ffn is not None and hasattr(ffn, "reset_expert_usage"):
                method = getattr(ffn, "reset_expert_usage", None)
                if callable(method):
                    method()
        
        for layer in self.decoder_layers:
            ffn = getattr(layer, "ffn", None)
            if ffn is not None and hasattr(ffn, "reset_expert_usage"):
                method = getattr(ffn, "reset_expert_usage", None)
                if callable(method):
                    method()