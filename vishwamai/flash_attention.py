"""
Re-export of FlashAttention functionality from vishwamai.layers.attention
This module exists for backward compatibility with code that imports from vishwamai.flash_attention
"""

# Re-export all flash attention-related components from layers.attention
from vishwamai.layers.attention import (
    FlashAttention,
    flash_attention,
    flash_attention_inference,
    create_flash_attention,
    mha_with_flash_attention,
)

__all__ = [
    'FlashAttention',
    'flash_attention',
    'flash_attention_inference',
    'create_flash_attention',
    'mha_with_flash_attention',
]