import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """
    LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False.
    """

    def __init__(self, ndim: int, has_bias: bool):
        """
        Initialize LayerNorm.
        
        Args:
            ndim: Normalized dimension size
            has_bias: Whether to include a bias parameter
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if has_bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization.
        
        Args:
            x: Input tensor of shape (..., ndim)
            
        Returns:
            Normalized tensor of same shape
        """
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)

