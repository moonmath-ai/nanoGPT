import torch
import torch.nn as nn

try:
    from .LayerNorm import LayerNorm
except ImportError:
    from LayerNorm import LayerNorm


class MLP(nn.Module):
    """
    Multi-layer perceptron (feed-forward network) for transformer blocks.
    
    Architecture: LayerNorm -> Linear -> GELU -> Linear -> Dropout -> residual
    
    Includes LayerNorm and residual connection internally.
    """

    def __init__(self, config):
        """
        Initialize MLP.
        
        Args:
            config: Configuration object with required parameters:
                - n_embd: Embedding dimension
                - has_bias: Whether to use bias in layers
                - dropout: Dropout rate
        """
        super().__init__()
        # Validate required config parameters
        assert hasattr(config, 'n_embd'), "config must have 'n_embd'"
        assert hasattr(config, 'has_bias'), "config must have 'has_bias'"
        assert hasattr(config, 'dropout'), "config must have 'dropout'"
        
        # Layer normalization for pre-norm architecture
        self.ln = LayerNorm(config.n_embd, has_bias=config.has_bias)
        
        # Feed-forward layers
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.has_bias)
        self.gelu = nn.GELU()
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.has_bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through MLP with LayerNorm and residual connection.
        
        Args:
            x: Input tensor of shape (B, seq_len, n_embd) where seq_len can be T, or q_len
            
        Returns:
            Output tensor of same shape as input with residual connection applied
        """
        # Pre-norm architecture: LayerNorm -> MLP -> residual connection
        residual = x
        x = self.ln(x)
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        x = x + residual
        return x

