"""Test suite for verifying VishwamAI imports and dependencies."""

import pytest

def test_jax_imports():
    """Test JAX and related imports."""
    import jax
    import jax.numpy as jnp
    
    # Force CPU device usage when no GPU is available
    jax.config.update('jax_platforms', 'cpu')
    
    devices = jax.devices()
    assert len(devices) > 0, "No JAX devices found"
    assert all(isinstance(d.platform, str) for d in devices)

def test_core_imports():
    """Test core VishwamAI imports."""
    from vishwamai.model import VishwamAI, VishwamAIConfig
    from vishwamai.transformer import EnhancedTransformerModel
    from vishwamai.device_mesh import TPUMeshContext
    from vishwamai.pipeline import TPUDataPipeline

def test_layer_imports():
    """Test layer imports."""
    from vishwamai.layers import (
        TPUMultiHeadAttention,
        TPUGEMMLinear,
        TPULayerNorm
    )

def test_kernel_imports():
    """Test kernel imports."""
    from vishwamai.kernels.core.kernel import (
        HardwareType,
        KernelConfig,
        act_quant,
        optimize_kernel_layout
    )

def test_config_imports():
    """Test configuration imports."""
    from vishwamai.configs.tpu_v3_config import TPUV3Config
    from vishwamai.configs.budget_model_config import BudgetModelConfig
