"""
VishwamAI Unified Attention Module
Provides a unified interface for attention mechanisms across different hardware accelerators (GPU/TPU)
with automatic device detection and optimization.
"""

import os
import torch
import jax
import jax.numpy as jnp
from typing import Optional, Union, Dict, Any
from enum import Enum

from vishwamai.models.gpu.attention import (
    BaseAttention as GPUBaseAttention,
    FlashMLAAttention as GPUFlashMLAAttention
)
from vishwamai.models.tpu.attention import (
    BaseAttention as TPUBaseAttention,
    FlashMLAttentionTPU
)

class DeviceType(Enum):
    """Enum for supported device types"""
    GPU = "gpu"
    TPU = "tpu"
    CPU = "cpu"

def detect_device() -> DeviceType:
    """Detect the available hardware accelerator"""
    if torch.cuda.is_available():
        return DeviceType.GPU
    try:
        # Check for TPU
        import jax.tools.colab_tpu
        jax.tools.colab_tpu.setup_tpu()
        return DeviceType.TPU
    except:
        return DeviceType.CPU

class AttentionFactory:
    """Factory class for creating appropriate attention mechanisms based on device"""
    
    @staticmethod
    def create_attention(
        attention_type: str,
        embed_dim: int,
        num_heads: int,
        device_type: Optional[DeviceType] = None,
        **kwargs
    ) -> Union[GPUBaseAttention, TPUBaseAttention]:
        """
        Create an attention instance based on device type and attention variant
        
        Args:
            attention_type: Type of attention mechanism ("base", "flash_mla")
            embed_dim: Embedding dimension
            num_heads: Number of attention heads
            device_type: Override automatic device detection
            **kwargs: Additional attention-specific parameters
        """
        if device_type is None:
            device_type = detect_device()

        if device_type == DeviceType.GPU:
            if attention_type == "base":
                return GPUBaseAttention(embed_dim, num_heads, **kwargs)
            elif attention_type == "flash_mla":
                return GPUFlashMLAAttention(embed_dim, num_heads, **kwargs)
            else:
                raise ValueError(f"Unsupported attention type: {attention_type}")

        elif device_type == DeviceType.TPU:
            if attention_type == "base":
                return TPUBaseAttention(embed_dim, num_heads, **kwargs)
            elif attention_type == "flash_mla":
                return FlashMLAttentionTPU(embed_dim, num_heads, **kwargs)
            else:
                raise ValueError(f"Unsupported attention type: {attention_type}")

        else:
            raise ValueError(f"Unsupported device type: {device_type}")

class UnifiedAttention:
    """
    Unified attention interface that automatically selects and initializes 
    the appropriate attention implementation based on available hardware.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        attention_type: str = "flash_mla",
        device_override: Optional[DeviceType] = None,
        **kwargs
    ):
        self.device_type = device_override or detect_device()
        self.attention = AttentionFactory.create_attention(
            attention_type=attention_type,
            embed_dim=embed_dim,
            num_heads=num_heads,
            device_type=self.device_type,
            **kwargs
        )
        
    def __call__(self, *args, **kwargs):
        """Forward pass delegation to underlying attention implementation"""
        return self.attention(*args, **kwargs)

    @property
    def device(self) -> DeviceType:
        """Get current device type"""
        return self.device_type

def create_attention(
    embed_dim: int,
    num_heads: int,
    attention_type: str = "flash_mla",
    device: Optional[str] = None,
    **kwargs
) -> UnifiedAttention:
    """
    Convenience function to create attention modules with automatic device selection
    
    Args:
        embed_dim: Embedding dimension
        num_heads: Number of attention heads
        attention_type: Type of attention mechanism ("base", "flash_mla")
        device: Optional device override ("gpu", "tpu", "cpu")
        **kwargs: Additional attention-specific parameters
    
    Returns:
        UnifiedAttention: Appropriate attention implementation for the device
    """
    device_override = DeviceType(device.lower()) if device else None
    return UnifiedAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        attention_type=attention_type,
        device_override=device_override,
        **kwargs
    )

# Example usage:
if __name__ == "__main__":
    # Create attention with automatic device detection
    attention = create_attention(
        embed_dim=512,
        num_heads=8,
        attention_type="flash_mla",
        dropout=0.1
    )
    
    print(f"Using attention on device: {attention.device}")

