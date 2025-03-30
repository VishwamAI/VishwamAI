"""Tests for Gemma 3 distillation kernel and layer optimizers."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from typing import Tuple, Dict

from vishwamai.kernels.tpu.distillation_kernels import (
    DistillationKernelConfig,
    TeacherStudentAttention,
    KernelManager,
    LayerwiseOptimizer
)
from vishwamai.kernels.tpu.layer_optimizers import (
    LayerOptConfig,
    AdaptiveLayerOptimizer,
    LayerNormOptimizer,
    FFNOptimizer,
    AdaptiveMoEOptimizer
)

@pytest.fixture
def kernel_config():
    """Create test kernel configuration."""
    return DistillationKernelConfig(
        block_size=128,
        use_flash_attention=True,
        use_fp8=True,
        dtype=jnp.bfloat16
    )

@pytest.fixture
def layer_config():
    """Create test layer configuration."""
    return LayerOptConfig(
        hidden_dim=768,
        num_heads=12,
        head_dim=64,
        mlp_dim=3072,
        dropout_rate=0.1,
        attention_dropout=0.1
    )

@pytest.fixture
def sample_attention_inputs(layer_config):
    """Create sample attention inputs for testing."""
    batch_size = 4
    seq_len = 256
    
    # Create random inputs
    rng = jax.random.PRNGKey(0)
    k1, k2, k3 = jax.random.split(rng, 3)
    
    shape = (batch_size, layer_config.num_heads, seq_len, layer_config.head_dim)
    
    teacher_q = jax.random.normal(k1, shape)
    teacher_k = jax.random.normal(k2, shape)
    teacher_v = jax.random.normal(k3, shape)
    
    # Create student inputs as slightly perturbed versions
    student_q = teacher_q + 0.1 * jax.random.normal(k1, shape)
    student_k = teacher_k + 0.1 * jax.random.normal(k2, shape)
    student_v = teacher_v + 0.1 * jax.random.normal(k3, shape)
    
    return {
        "teacher": (teacher_q, teacher_k, teacher_v),
        "student": (student_q, student_k, student_v)
    }

def test_teacher_student_attention(kernel_config, sample_attention_inputs):
    """Test TeacherStudentAttention implementation."""
    attention = TeacherStudentAttention(kernel_config)
    
    teacher_inputs = sample_attention_inputs["teacher"]
    student_inputs = sample_attention_inputs["student"]
    
    # Test without intermediates
    output = attention.transfer_attention_maps(
        *teacher_inputs,
        *student_inputs
    )
    assert isinstance(output, jnp.ndarray)
    assert output.dtype == kernel_config.dtype
    
    # Test with intermediates
    output, intermediates = attention.transfer_attention_maps(
        *teacher_inputs,
        *student_inputs,
        return_intermediates=True
    )
    
    assert isinstance(intermediates, dict)
    assert "transfer_loss" in intermediates
    assert intermediates["transfer_loss"].ndim > 0

def test_kernel_manager(kernel_config, layer_config):
    """Test KernelManager functionality."""
    manager = KernelManager(kernel_config)
    
    # Test layout optimization
    x = jnp.ones((4, 12, 256, 64))
    optimized = manager.optimize_kernel_layout(x)
    assert optimized.shape == x.shape
    
    # Test operation fusion
    inputs = {
        "x": jnp.ones((4, 256, 768))
    }
    
    def op1(x):
        return x * 2
    
    def op2(x):
        return x + 1
    
    ops = {
        "first": op1,
        "second": (op2, ["first"])
    }
    
    outputs = manager.fuse_operations(ops, inputs)
    assert "second" in outputs
    assert outputs["second"].shape == inputs["x"].shape

def test_layerwise_optimizer(kernel_config, layer_config):
    """Test LayerwiseOptimizer functionality."""
    optimizer = LayerwiseOptimizer(
        kernel_config,
        num_layers=4,
        dropout_rate=0.1
    )
    
    # Test progressive dropout
    for i in range(4):
        rate = optimizer.get_layer_dropout(i)
        assert 0.1 <= rate <= 0.2  # Progressive increase

def test_adaptive_layer_optimizer(kernel_config, layer_config):
    """Test AdaptiveLayerOptimizer functionality."""
    optimizer = AdaptiveLayerOptimizer(layer_config, kernel_config)
    
    # Mock layer and inputs
    class MockLayer:
        def feed_forward(self, x):
            return x
        def layer_norm(self, x):
            return x
    
    teacher_layer = MockLayer()
    student_layer = MockLayer()
    
    inputs = {
        "query": jnp.ones((4, 12, 256, 64)),
        "key": jnp.ones((4, 12, 256, 64)),
        "value": jnp.ones((4, 12, 256, 64))
    }
    
    outputs, stats = optimizer.optimize_attention_layer(
        teacher_layer,
        student_layer,
        inputs
    )
    
    assert isinstance(outputs, dict)
    assert isinstance(stats, dict)
    assert "attention_similarity" in stats

def test_layer_norm_optimizer(layer_config):
    """Test LayerNormOptimizer functionality."""
    norm = LayerNormOptimizer(layer_config)
    
    x = jnp.ones((4, 256, 768))
    scale = jnp.ones(768)
    bias = jnp.zeros(768)
    
    output = norm(x, scale, bias)
    assert output.shape == x.shape
    assert output.dtype == layer_config.dtype
    
    # Test normalization properties
    mean = jnp.mean(output, axis=-1)
    var = jnp.var(output, axis=-1)
    np.testing.assert_allclose(mean, 0.0, atol=1e-6)
    np.testing.assert_allclose(var, 1.0, atol=1e-6)

def test_ffn_optimizer(layer_config):
    """Test FFNOptimizer functionality."""
    ffn = FFNOptimizer(layer_config)
    
    # Create sample inputs and weights
    x = jnp.ones((4, 256, 768))
    wi = jnp.ones((768, 3072))
    wo = jnp.ones((3072, 768))
    
    # Test forward pass
    output = ffn(x, wi, wo)
    assert output.shape == x.shape
    
    # Test with dropout
    rng = jax.random.PRNGKey(0)
    output_dropout = ffn(x, wi, wo, dropout_rng=rng)
    assert output_dropout.shape == x.shape

def test_adaptive_moe_optimizer(layer_config):
    """Test AdaptiveMoEOptimizer functionality."""
    moe = AdaptiveMoEOptimizer(
        layer_config,
        num_experts=4
    )
    
    # Create sample inputs
    batch_size = 4
    seq_len = 256
    hidden_dim = 768
    expert_dim = 3072
    
    x = jnp.ones((batch_size, seq_len, hidden_dim))
    router_weights = jnp.ones((hidden_dim, moe.num_experts))
    
    # Create expert weights
    expert_weights = [
        (jnp.ones((hidden_dim, expert_dim)),
         jnp.ones((expert_dim, hidden_dim)))
        for _ in range(moe.num_experts)
    ]
    
    # Test routing and computation
    output, aux = moe.route_and_compute(
        x,
        router_weights,
        expert_weights
    )
    
    assert output.shape == x.shape
    assert "router_probs" in aux
    assert "aux_loss" in aux
    assert aux["router_probs"].shape == (batch_size, seq_len, moe.num_experts)

def test_end_to_end_distillation(
    kernel_config,
    layer_config,
    sample_attention_inputs
):
    """Test end-to-end distillation optimization."""
    # Initialize components
    attention = TeacherStudentAttention(kernel_config)
    layer_opt = AdaptiveLayerOptimizer(layer_config, kernel_config)
    norm = LayerNormOptimizer(layer_config)
    ffn = FFNOptimizer(layer_config)
    
    # Process attention
    teacher_inputs = sample_attention_inputs["teacher"]
    student_inputs = sample_attention_inputs["student"]
    
    attn_out, intermediates = attention.transfer_attention_maps(
        *teacher_inputs,
        *student_inputs,
        return_intermediates=True
    )
    
    # Verify attention output
    assert attn_out.dtype == kernel_config.dtype
    assert "transfer_loss" in intermediates
    
    # Process through layer optimization
    class MockLayer:
        def feed_forward(self, x):
            return x
        def layer_norm(self, x):
            return x
    
    teacher_layer = MockLayer()
    student_layer = MockLayer()
    
    layer_out, stats = layer_opt.optimize_attention_layer(
        teacher_layer,
        student_layer,
        {"attention": attn_out}
    )
    
    # Verify layer optimization
    assert "attention_similarity" in stats
    assert isinstance(layer_out, dict)
    
    # Test full pipeline with multiple components
    assert jnp.isfinite(stats["attention_similarity"])
    if "grad_norm_ratio" in stats:
        assert jnp.isfinite(stats["grad_norm_ratio"])
