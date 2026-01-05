import torch
import torch.nn as nn
from typing import Optional

try:
    from .Attention import Attention
    from .MLP import MLP
except ImportError:
    from Attention import Attention
    from MLP import MLP


class Transformer(nn.Module):
    """
    Transformer block with pre-norm architecture.
    
    Architecture:
        x -> Attention (includes LayerNorm + residual) -> MLP (includes LayerNorm + residual)
    
    Can be configured with different attention types:
    - full_self_attn (default): Bidirectional self-attention
    - causal_self_attn: Causal (autoregressive) self-attention
    - cross_attn: Cross-attention with separate k,v source
    - trunc_self_attn: Self-attention with truncated queries
    - causal_trunc_self_attn: Causal self-attention with truncated queries
    - latent_attn: Attention with learned latent queries
    """

    def __init__(self, config, attn_type: str = 'full_self_attn'):
        """
        Initialize transformer block.
        
        Args:
            config: Configuration object with required parameters:
                - n_embd: Embedding dimension
                - has_bias: Whether to use bias in layers
                - dropout: Dropout rate
                - n_head: Number of attention heads
                - latent_q_len: Query length (for latent_attn mode only, required)
                - latent_init_std: Standard deviation for latent init (for latent_attn mode only, required)
            attn_type: Type of attention ('full_self_attn', 'causal_self_attn', 'cross_attn', 'trunc_self_attn', 'causal_trunc_self_attn', or 'latent_attn')
        """
        super().__init__()
        # Validate required config parameters
        assert hasattr(config, 'n_embd'), "config must have 'n_embd'"
        assert hasattr(config, 'has_bias'), "config must have 'has_bias'"
        assert hasattr(config, 'dropout'), "config must have 'dropout'"
        assert hasattr(config, 'n_head'), "config must have 'n_head'"
        
        self.attn_type = attn_type
        self.attn = Attention(config, attn_type=attn_type)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor, q_len: Optional[int] = None, y: Optional[torch.Tensor] = None, rope_start_idx: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through transformer block.
        
        Args:
            x: Input tensor of shape (B, T, n_embd)
            q_len: Output length for trunc_self_attn/causal_trunc_self_attn (required for those modes)
            y: Context tensor of shape (B, S, n_embd) for cross_attn mode (required for cross_attn)
            rope_start_idx: Starting position index for RoPE (default None, skips RoPE if None)
            
        Returns:
            Output tensor of shape (B, T, n_embd) for full/causal_self_attn and cross_attn,
            (B, q_len, n_embd) for trunc_self_attn, causal_trunc_self_attn, and latent_attn
        """
        x = self.attn(x, q_len=q_len, y=y, rope_start_idx=rope_start_idx)
        x = self.mlp(x)
        return x