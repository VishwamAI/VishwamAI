"""
Layer normalization implementation optimized for T4 GPUs
"""
import torch
import torch.nn as nn

class T4LayerNorm(nn.Module):
    """
    Layer normalization optimized for T4 GPUs with FP16/BF16 support
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply layer normalization to input tensor
        
        Args:
            hidden_states: Input tensor of shape [..., hidden_size]
            
        Returns:
            Normalized tensor of the same shape
        """
        # Compute mean and variance
        mean = hidden_states.mean(-1, keepdim=True)
        variance = (hidden_states - mean).pow(2).mean(-1, keepdim=True)
        
        # Normalize and scale
        hidden_states = (hidden_states - mean) / torch.sqrt(variance + self.variance_epsilon)
        hidden_states = hidden_states * self.weight + self.bias
        
        return hidden_states

class RMSNorm(nn.Module):
    """
    Root Mean Square Layer Normalization
    """
    def __init__(self, hidden_size: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply RMS normalization to input tensor
        
        Args:
            hidden_states: Input tensor of shape [..., hidden_size]
            
        Returns:
            Normalized tensor of the same shape
        """
        # Compute RMS
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        
        # Scale
        return self.weight * hidden_states
