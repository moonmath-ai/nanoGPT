import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

try:
    from .LayerNorm import LayerNorm
except ImportError:
    from LayerNorm import LayerNorm


class Attention(nn.Module):
    """
    Attention with RoPE (Rotary Position Embedding) and residual connection.
    All RoPE functions are included as class methods for simplicity.
    
    Architecture: LayerNorm -> Attention -> residual connection
    
    Can be configured as:
    - full_self_attn (default): bidirectional, q, k, v are projections of input x, residual is x
    - causal_self_attn: causal (autoregressive), q, k, v are projections of input x, residual is x
    - cross_attn: bidirectional, q from y, k, v from x, residual is y
    - causal_cross_attn: causal, q from y, k, v from x, residual is y (requires len(y) <= len(x))
    - trunc_self_attn: bidirectional, q from last q_len of x, k, v from full x, RoPE applied accordingly
    - causal_trunc_self_attn: causal, q from last q_len of x, k, v from full x, RoPE applied accordingly
    - latent_attn: bidirectional, q is learned latent, k, v from input x, residual is q
    """
    
    # Class-level cache for freqs_cis (always starts from idx=0)
    _freqs_cis = torch.empty(0, dtype=torch.complex64)
    
    def __init__(self, config, attn_type: str = 'full_self_attn'):
        """
        Initialize attention module.
        
        Args:
            config: Configuration object with required parameters:
                - n_embd: Embedding dimension
                - n_head: Number of attention heads
                - has_bias: Whether to use bias in layers
                - dropout: Dropout rate
                - latent_q_len: Query length (for latent_attn mode only, required)
                - latent_init_std: Standard deviation for latent init (for latent_attn mode only, required)
            attn_type: Type of attention ('full_self_attn', 'causal_self_attn', 'cross_attn', 'causal_cross_attn', 'trunc_self_attn', 'causal_trunc_self_attn', or 'latent_attn')
        """
        super().__init__()
        # Validate required config parameters
        assert config.n_embd % config.n_head == 0
        valid_types = ['full_self_attn', 'causal_self_attn', 'cross_attn', 'causal_cross_attn', 'trunc_self_attn', 'causal_trunc_self_attn', 'latent_attn']
        assert attn_type in valid_types, f"attn_type must be one of {valid_types}, got {attn_type}"
        
        self.attn_type = attn_type
        
        # Layer normalization for pre-norm architecture
        self.ln = LayerNorm(config.n_embd, has_bias=config.has_bias)
        
        # Query projection or learned latent
        if attn_type in ['full_self_attn', 'causal_self_attn', 'cross_attn', 'causal_cross_attn', 'trunc_self_attn', 'causal_trunc_self_attn']:
            self.c_q = nn.Linear(config.n_embd, config.n_embd, bias=config.has_bias)
        else:  # latent_attn
            assert hasattr(config, 'latent_q_len'), "config must have 'latent_q_len' for latent_attn mode"
            assert hasattr(config, 'latent_init_std'), "config must have 'latent_init_std' for latent_attn mode"
            self.c_q = nn.Parameter(torch.randn(1, config.latent_q_len, config.n_embd) * config.latent_init_std)
        
        # Key, value, and output projections
        self.c_k = nn.Linear(config.n_embd, config.n_embd, bias=config.has_bias)
        self.c_v = nn.Linear(config.n_embd, config.n_embd, bias=config.has_bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.has_bias)
        
        # Dropout
        self.dropout_fun = nn.Dropout(config.dropout)
        
        # Store config values
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = config.n_embd // config.n_head
        self.dropout = config.dropout
        self.latent_q_len = config.latent_q_len if attn_type == 'latent_attn' else None
        
        # Require flash attention (PyTorch >= 2.0)
        if not hasattr(torch.nn.functional, 'scaled_dot_product_attention'):
            raise RuntimeError("Flash Attention (scaled_dot_product_attention) is required but not available. "
                             "Please use PyTorch >= 2.0.")
    
    def precompute_freqs_cis(self, seq_len: int, theta: float = 10000.0, rope_start_idx: int = 0) -> torch.Tensor:
        """
        Precompute RoPE frequency tensor with caching.
        The cache always starts from idx=0 and extends as needed.
        
        Args:
            seq_len: Maximum sequence length
            theta: Base frequency (default 10000.0)
            rope_start_idx: Starting position index (default 0) - used for slicing the cache
            
        Returns:
            freqs_cis: (seq_len, head_dim // 2) complex tensor
        """
        # Determine the total length needed (from idx=0)
        total_len_needed = rope_start_idx + seq_len
        
        # If cache has enough values, return immediately
        if total_len_needed <= Attention._freqs_cis.shape[0]:
            return Attention._freqs_cis[rope_start_idx:rope_start_idx + seq_len]
        
        # Extend the cache: compute only the new frequencies needed
        freqs = 1.0 / (theta ** (torch.arange(0, self.head_dim, 2)[: (self.head_dim // 2)].float() / self.head_dim))
        current_len = Attention._freqs_cis.shape[0]
        t = torch.arange(current_len, total_len_needed, dtype=torch.float32)  # (new_len,)
        freqs_outer = torch.outer(t, freqs)  # (new_len, head_dim // 2)
        new_freqs_cis = torch.polar(torch.ones_like(freqs_outer), freqs_outer)  # complex64: (new_len, head_dim // 2)
        Attention._freqs_cis = torch.cat([Attention._freqs_cis, new_freqs_cis], dim=0)
        
        # Return the requested slice: from rope_start_idx to rope_start_idx + seq_len
        return Attention._freqs_cis[rope_start_idx:rope_start_idx + seq_len]
    
    def apply_rotary_emb(self, x: torch.Tensor, rope_start_idx: Optional[int] = None) -> torch.Tensor:
        """
        Apply rotary embeddings to a single tensor (query or key).
        
        Args:
            x: (B, T, nh, head_dim) query or key tensor
            rope_start_idx: Starting position index (default None, skips RoPE if None)
            
        Returns:
            x_out: (B, T, nh, head_dim) rotated tensor (or unchanged if rope_start_idx is None)
        """
        # Skip RoPE if rope_start_idx is None
        if rope_start_idx is None:
            return x
        
        # Get sequence length from input tensor
        seq_len = x.shape[1]
        
        # Precompute frequencies (uses cached values) and move to input device
        freqs_cis = self.precompute_freqs_cis(seq_len, rope_start_idx=rope_start_idx).to(x.device)  # (T, head_dim // 2)
        
        # Reshape to complex: (B, T, nh, head_dim // 2, 2) -> (B, T, nh, head_dim // 2) complex
        x_ = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
        
        # Reshape freqs_cis for broadcasting: (T, head_dim // 2) -> (1, T, 1, head_dim // 2)
        freqs_cis = freqs_cis.view(1, x_.shape[1], 1, x_.shape[-1])
        
        # Broadcast and apply rotation: (B, T, nh, head_dim)
        x_out = torch.view_as_real(x_ * freqs_cis).flatten(3)
        
        return x_out.type_as(x)
    
    def forward(self, x: torch.Tensor, q_len: Optional[int] = None, y: Optional[torch.Tensor] = None, rope_start_idx: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass through attention with RoPE, LayerNorm, and residual connection.
        
        Args:
            x: (B, T, n_embd) input tensor
                - For full/causal_self_attn: used for q, k, v
                - For cross_attn/causal_cross_attn: used for k, v only
                - For trunc_self_attn/causal_trunc_self_attn: last q_len positions used for q, full x for k, v
                - For latent_attn: used for k, v only
            q_len: Output length for trunc_self_attn/causal_trunc_self_attn (required for those modes)
            y: (B, S, n_embd) tensor for cross_attn/causal_cross_attn (q source)
            rope_start_idx: Starting position index for RoPE (default None, skips RoPE if None)
            
        Returns:
            Output tensor: (B, T, n_embd) for full/causal_self_attn; (B, q_len, n_embd) for cross_attn, causal_cross_attn, trunc_self_attn, causal_trunc_self_attn, and latent_attn
        """
        B, kv_len, n_embd = x.size()  # batch, sequence length, embedding dim
        assert n_embd == self.n_embd, "Input embedding dimension must match model dimension"
        
        # Validate y for cross_attn and causal_cross_attn
        if self.attn_type in ['cross_attn', 'causal_cross_attn']:
            assert y is not None, "y is required for cross_attn/causal_cross_attn mode"
            assert y.size(2) == self.n_embd, "y embedding dimension must match model dimension"
            q_len = y.size(1)  # q comes from y
            if self.attn_type == 'causal_cross_attn':
                assert q_len <= kv_len, f"causal_cross_attn requires len(y) ({q_len}) <= len(x) ({kv_len})"
        
        # Validate q_len for trunc modes
        if self.attn_type in ['trunc_self_attn', 'causal_trunc_self_attn']:
            assert q_len is not None, "q_len is required for trunc_self_attn/causal_trunc_self_attn mode"
            assert q_len <= kv_len, f"q_len ({q_len}) must be <= kv_len ({kv_len})"
        
        # Store residual for pre-norm architecture
        if self.attn_type in ['full_self_attn', 'causal_self_attn']:
            residual = x  # (B, kv_len, n_embd)
        elif self.attn_type in ['cross_attn', 'causal_cross_attn']:
            residual = y  # (B, q_len, n_embd)
        elif self.attn_type in ['trunc_self_attn', 'causal_trunc_self_attn']:
            residual = x[:, -q_len:, :]  # (B, q_len, n_embd)
        else:  # latent_attn
            residual = self.c_q.expand(B, -1, -1)  # (B, latent_q_len, n_embd)
        
        # Apply layer normalization
        x_norm = self.ln(x)  # (B, kv_len, n_embd)
        
        # Compute queries
        if self.attn_type in ['full_self_attn', 'causal_self_attn']:
            q = self.c_q(x_norm)  # (B, kv_len, n_embd)
            q_len = kv_len
        elif self.attn_type in ['cross_attn', 'causal_cross_attn']:
            q = self.c_q(self.ln(y))  # (B, q_len, n_embd)
            # q_len already set from y.size(1)
        elif self.attn_type in ['trunc_self_attn', 'causal_trunc_self_attn']:
            q = self.c_q(x_norm[:, -q_len:, :])  # (B, q_len, n_embd)
            # q_len already set from argument
        else:  # latent_attn
            q = self.c_q.expand(B, -1, -1)  # (B, latent_q_len, n_embd)
            q_len = self.latent_q_len
        
        # Get k, v source
        if self.attn_type in ['cross_attn', 'causal_cross_attn']:
            kv_source = x_norm  # (B, kv_len, n_embd) - k, v from x
            # kv_len already set from x.size()
        else:
            kv_source = x_norm  # (B, kv_len, n_embd)
            # kv_len already set from x.size()
        
        # Compute keys and values
        k = self.c_k(kv_source)  # (B, kv_len, n_embd)
        v = self.c_v(kv_source)  # (B, kv_len, n_embd)
        
        # Reshape for multi-head attention: (B, seq, n_embd) -> (B, seq, nh, hd)
        q = q.view(B, q_len, self.n_head, self.head_dim)  # (B, q_len, nh, hd)
        k = k.view(B, kv_len, self.n_head, self.head_dim)  # (B, kv_len, nh, hd)
        v = v.view(B, kv_len, self.n_head, self.head_dim)  # (B, kv_len, nh, hd)
        
        # Apply RoPE if rope_start_idx is provided
        # q offset accounts for truncation: when kv_len == q_len, offset is rope_start_idx + 0
        if rope_start_idx is not None:
            q = self.apply_rotary_emb(q, rope_start_idx=rope_start_idx + (kv_len - q_len))  # (B, q_len, nh, hd)
            k = self.apply_rotary_emb(k, rope_start_idx=rope_start_idx)  # (B, kv_len, nh, hd)
        
        # Transpose for attention: (B, seq, nh, hd) -> (B, nh, seq, hd)
        q = q.transpose(1, 2)  # (B, nh, q_len, hd)
        k = k.transpose(1, 2)  # (B, nh, kv_len, hd)
        v = v.transpose(1, 2)  # (B, nh, kv_len, hd)
        
        # Build attention mask for causal modes with q_len != kv_len
        # Queries are aligned with END of kv sequence: query i corresponds to position (kv_len-q_len+i)
        attn_mask = None
        if self.attn_type in ['causal_trunc_self_attn', 'causal_cross_attn']:
            # Create mask: (q_len, kv_len) where mask[i,j] = True if j <= (kv_len-q_len)+i
            q_positions = torch.arange(kv_len - q_len, kv_len, device=x.device)  # (q_len,)
            k_positions = torch.arange(kv_len, device=x.device)  # (kv_len,)
            attn_mask = k_positions.unsqueeze(0) <= q_positions.unsqueeze(1)  # (q_len, kv_len)
        
        # Scaled dot-product attention (is_causal only for causal_self_attn; others use attn_mask)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attn_mask,
            dropout_p=self.dropout if self.training else 0,
            is_causal=(self.attn_type == 'causal_self_attn')
        )  # (B, nh, q_len, hd)
        
        # Reshape back: (B, nh, q_len, hd) -> (B, q_len, n_embd)
        out = out.transpose(1, 2).contiguous().view(B, q_len, n_embd)  # (B, q_len, n_embd)
        
        # Output projection and dropout
        out = self.c_proj(out)  # (B, q_len, n_embd)
        out = self.dropout_fun(out)  # (B, q_len, n_embd)
        
        # Residual connection
        out = out + residual  # (B, q_len, n_embd)
        
        return out

