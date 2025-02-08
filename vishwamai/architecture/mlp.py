"""
Memory-Efficient MLP Implementation
================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast
from typing import Optional, Union, Literal

class MemoryEfficientMLP(nn.Module):
    """Memory-efficient MLP with optional quantization and chunked computation."""
    def __init__(
        self, 
        dim: int, 
        hidden_dim: int, 
        dropout: float = 0.1,
        quantization: Optional[Literal["int8", "fp8", None]] = None,
        chunk_size: int = 128
    ):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        self.chunk_size = chunk_size
        self.quantization = quantization
        
        if quantization == "int8":
            self.fc1 = torch.quantization.quantize_dynamic(
                self.fc1, {torch.nn.Linear}, dtype=torch.qint8
            )
            self.fc2 = torch.quantization.quantize_dynamic(
                self.fc2, {torch.nn.Linear}, dtype=torch.qint8
            )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        chunks = []
        
        # Process input in chunks for memory efficiency
        for i in range(0, L, self.chunk_size):
            chunk = x[:, i:min(i + self.chunk_size, L)]
            with autocast(enabled=True):
                # First linear layer + activation
                chunk = self.fc1(chunk)
                chunk = self.act(chunk)
                chunk = self.drop(chunk)
                
                # Second linear layer
                chunk = self.fc2(chunk)
                chunk = self.drop(chunk)
            chunks.append(chunk)
        
        return torch.cat(chunks, dim=1)

class MemoryEfficientFFN(nn.Module):
    """Feed-Forward Network with configurable activation."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
        
        if activation == "gelu":
            self.act = nn.GELU()
        elif activation == "relu":
            self.act = nn.ReLU()
        elif activation == "silu":
            self.act = nn.SiLU()
        else:
            raise ValueError(f"Unsupported activation: {activation}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class MemoryEfficientGatedMLP(nn.Module):
    """Gated MLP with additional control mechanism."""
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Linear(dim, hidden_dim)
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, L, D = x.shape
        chunks = []
        
        for i in range(0, L, self.chunk_size):
            chunk = x[:, i:min(i + self.chunk_size, L)]
            with autocast(enabled=True):
                # Compute gate and transform in parallel
                gate = torch.sigmoid(self.gate(chunk))
                transformed = self.fc1(chunk)
                transformed = self.act(transformed)
                
                # Fused operations for better performance
                chunk = transformed * gate
                chunk = self.drop(chunk)
                chunk = self.fc2(chunk)
                chunk = self.drop(chunk)
            chunks.append(chunk)
        
        return torch.cat(chunks, dim=1)

# Aliases for backward compatibility
MLP = MemoryEfficientMLP
FFN = MemoryEfficientFFN
GatedMLP = MemoryEfficientGatedMLP
