import torch
import torch.nn as nn
import math
from typing import Optional, Tuple
from .moe_layer import SparseMoELayer


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding"""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoderLayerWithMoE(nn.Module):
    """Transformer encoder layer with MoE replacing FFN"""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_experts: int,
        expert_hidden_dim: int,
        top_k: int,
        router_type: str = "top_k",
        dropout: float = 0.1,
        use_load_balancer: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe = SparseMoELayer(
            d_model, num_experts, expert_hidden_dim, top_k, 
            router_type, dropout, use_load_balancer
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        src: torch.Tensor, 
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self attention
        src2, _ = self.self_attn(src, src, src, attn_mask=src_mask, 
                                  key_padding_mask=src_key_padding_mask)
        src = src + self.dropout(src2)
        src = self.norm1(src)
        
        # MoE
        src2, aux_loss = self.moe(src)
        src = src + self.dropout(src2)
        src = self.norm2(src)
        
        return src, aux_loss


class TransformerDecoderLayerWithMoE(nn.Module):
    """Transformer decoder layer with MoE replacing FFN"""
    
    def __init__(
        self,
        d_model: int,
        nhead: int,
        num_experts: int,
        expert_hidden_dim: int,
        top_k: int,
        router_type: str = "top_k",
        dropout: float = 0.1,
        use_load_balancer: bool = False,
    ):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.moe = SparseMoELayer(
            d_model, num_experts, expert_hidden_dim, top_k,
            router_type, dropout, use_load_balancer
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self attention
        tgt2, _ = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask,
                                  key_padding_mask=tgt_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm1(tgt)
        
        # Cross attention
        tgt2, _ = self.cross_attn(tgt, memory, memory, attn_mask=memory_mask,
                                   key_padding_mask=memory_key_padding_mask)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm2(tgt)
        
        # MoE
        tgt2, aux_loss = self.moe(tgt)
        tgt = tgt + self.dropout(tgt2)
        tgt = self.norm3(tgt)
        
        return tgt, aux_loss


class MoETransformer(nn.Module):
    """Full Encoder-Decoder Transformer with MoE layers"""
    
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        num_experts: int = 8,
        expert_hidden_dim: int = 2048,
        top_k: int = 2,
        router_type: str = "top_k",
        dropout: float = 0.1,
        max_seq_length: int = 512,
        use_load_balancer: bool = False,
        pad_token_id: int = 1,
    ):
        super().__init__()
        self.d_model = d_model
        self.pad_token_id = pad_token_id
        self.router_type = router_type
        
        # Embeddings
        self.encoder_embedding = nn.Embedding(vocab_size, d_model)
        self.decoder_embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_seq_length, dropout)
        
        # Encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayerWithMoE(
                d_model, nhead, num_experts, expert_hidden_dim, top_k,
                router_type, dropout, use_load_balancer
            )
            for _ in range(num_encoder_layers)
        ])
        
        # Decoder layers
        self.decoder_layers = nn.ModuleList([
            TransformerDecoderLayerWithMoE(
                d_model, nhead, num_experts, expert_hidden_dim, top_k,
                router_type, dropout, use_load_balancer
            )
            for _ in range(num_decoder_layers)
        ])
        
        # Output projection
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize parameters"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    
    def forward(
        self,
        src: torch.Tensor,
        tgt: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        tgt_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Args:
            src: (batch_size, src_seq_len)
            tgt: (batch_size, tgt_seq_len)
        Returns:
            logits: (batch_size, tgt_seq_len, vocab_size)
            aux_loss: Load balancing loss
        """
        # Encode
        memory, encoder_aux_loss = self.encode(src, src_mask, src_key_padding_mask)
        
        # Decode
        output, decoder_aux_loss = self.decode(
            tgt, memory, tgt_mask, None,
            tgt_key_padding_mask, src_key_padding_mask
        )
        
        # Combine auxiliary losses
        aux_loss = None
        if encoder_aux_loss is not None or decoder_aux_loss is not None:
            aux_loss = 0.0
            if encoder_aux_loss is not None:
                aux_loss += encoder_aux_loss
            if decoder_aux_loss is not None:
                aux_loss += decoder_aux_loss
        
        return output, aux_loss
    
    def encode(
        self,
        src: torch.Tensor,
        src_mask: Optional[torch.Tensor] = None,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Encode source sequence"""
        # Embed and add positional encoding
        src_emb = self.encoder_embedding(src) * math.sqrt(self.d_model)
        src_emb = self.pos_encoder(src_emb)
        
        # Pass through encoder layers
        total_aux_loss = 0.0
        aux_loss_count = 0
        output = src_emb
        
        for layer in self.encoder_layers:
            output, aux_loss = layer(output, src_mask, src_key_padding_mask)
            if aux_loss is not None:
                total_aux_loss += aux_loss
                aux_loss_count += 1
        
        avg_aux_loss = total_aux_loss / aux_loss_count if aux_loss_count > 0 else None
        return output, avg_aux_loss
    
    def decode(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None,
        tgt_key_padding_mask: Optional[torch.Tensor] = None,
        memory_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Decode target sequence"""
        # Embed and add positional encoding
        tgt_emb = self.decoder_embedding(tgt) * math.sqrt(self.d_model)
        tgt_emb = self.pos_encoder(tgt_emb)
        
        # Pass through decoder layers
        total_aux_loss = 0.0
        aux_loss_count = 0
        output = tgt_emb
        
        for layer in self.decoder_layers:
            output, aux_loss = layer(
                output, memory, tgt_mask, memory_mask,
                tgt_key_padding_mask, memory_key_padding_mask
            )
            if aux_loss is not None:
                total_aux_loss += aux_loss
                aux_loss_count += 1
        
        # Project to vocabulary
        logits = self.output_projection(output)
        
        avg_aux_loss = total_aux_loss / aux_loss_count if aux_loss_count > 0 else None
        return logits, avg_aux_loss
    
    def generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask for decoder"""
        mask = torch.triu(torch.ones(sz, sz), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def get_all_expert_usage_stats(self):
        """Get expert usage statistics from all layers"""
        stats = {
            'encoder': [],
            'decoder': []
        }
        
        for i, layer in enumerate(self.encoder_layers):
            stats['encoder'].append({
                'layer': i,
                'usage': layer.moe.get_expert_usage_stats().cpu().tolist()
            })
        
        for i, layer in enumerate(self.decoder_layers):
            stats['decoder'].append({
                'layer': i,
                'usage': layer.moe.get_expert_usage_stats().cpu().tolist()
            })
        
        return stats
    
    def reset_all_expert_usage_stats(self):
        """Reset expert usage statistics in all layers"""
        for layer in self.encoder_layers:
            layer.moe.reset_expert_usage_stats()
        for layer in self.decoder_layers:
            layer.moe.reset_expert_usage_stats()