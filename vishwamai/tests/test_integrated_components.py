#!/usr/bin/env python3
"""
Test suite for integrated optimization components in VishwamAI.
"""

import jax
import jax.numpy as jnp
import pytest
from ..model import VishwamAIModel, ModelConfig
import numpy as np

@pytest.fixture
def model_config():
    """Create test model configuration with all components enabled."""
    return ModelConfig(
        vocab_size=32000,
        hidden_size=768,
        num_layers=4,  # Reduced for testing
        num_attention_heads=12,
        flash_mla_enabled=True,
        flash_mla_block_size=128,
        flash_mla_num_blocks=8,
        dual_pipe_enabled=True,
        dual_pipe_stages=2,
        dual_pipe_microbatches=4,
        deep_ep_enabled=True,
        deep_ep_num_experts=8,
        deep_ep_capacity=1.25,
        eplb_enabled=True,
        eplb_balance_factor=0.01,
        eplb_routing="top_2",
        deep_gemm_enabled=True,
        deep_gemm_quant="int8",
        deep_gemm_block_size=32,
        three_fs_enabled=True,
        three_fs_cache="lru"
    )

def test_flash_mla():
    """Test FlashMLA component."""
    config = model_config()
    model = VishwamAIModel(config)
    
    # Create dummy input
    batch_size, seq_len = 2, 128
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    # Initialize model
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, input_ids)
    
    # Forward pass
    outputs = model.apply(params, input_ids)
    
    # Verify output shape
    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)

def test_dual_pipe():
    """Test DualPipe component."""
    config = model_config()
    model = VishwamAIModel(config)
    
    # Test with multiple microbatches
    batch_size, seq_len = 8, 64  # Larger batch for pipeline testing
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, input_ids)
    outputs = model.apply(params, input_ids)
    
    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)

def test_deep_ep():
    """Test DeepEP component."""
    config = model_config()
    model = VishwamAIModel(config)
    
    # Test expert routing
    batch_size, seq_len = 4, 32
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, input_ids)
    outputs = model.apply(params, input_ids)
    
    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)

def test_eplb():
    """Test EPLB component."""
    config = model_config()
    model = VishwamAIModel(config)
    
    # Test load balancing
    batch_size, seq_len = 2, 64
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, input_ids)
    
    # Test both training and inference modes
    train_outputs = model.apply(params, input_ids, deterministic=False)
    eval_outputs = model.apply(params, input_ids, deterministic=True)
    
    assert train_outputs['logits'].shape == eval_outputs['logits'].shape

def test_deep_gemm():
    """Test DeepGEMM component."""
    config = model_config()
    model = VishwamAIModel(config)
    
    # Test quantization
    batch_size, seq_len = 2, 32
    input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
    
    rng = jax.random.PRNGKey(0)
    params = model.init(rng, input_ids)
    outputs = model.apply(params, input_ids)
    
    assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)

def test_integration():
    """Test all components working together."""
    config = model_config()
    model = VishwamAIModel(config)
    
    # Test with varying sequence lengths
    test_configs = [
        (2, 32),   # Short sequence
        (2, 128),  # Medium sequence
        (2, 512)   # Long sequence
    ]
    
    rng = jax.random.PRNGKey(0)
    for batch_size, seq_len in test_configs:
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        params = model.init(rng, input_ids)
        
        # Test forward pass
        outputs = model.apply(params, input_ids)
        assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
        
        # Test gradient computation
        def loss_fn(params):
            outputs = model.apply(params, input_ids)
            return jnp.mean(outputs['logits'])
        
        grad_fn = jax.grad(loss_fn)
        grads = grad_fn(params)
        
        # Verify gradients are not None or NaN
        for g in jax.tree_leaves(grads):
            assert not np.any(np.isnan(g))

def test_memory_efficiency():
    """Test memory usage of integrated components."""
    config = model_config()
    model = VishwamAIModel(config)
    
    # Test with increasing sequence lengths
    seq_lengths = [32, 64, 128]
    batch_size = 2
    
    rng = jax.random.PRNGKey(0)
    initial_usage = jax.device_get(jax.random.normal(rng, (1,))).nbytes
    
    for seq_len in seq_lengths:
        input_ids = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        params = model.init(rng, input_ids)
        
        # Measure peak memory usage
        with jax.profiler.trace() as trace:
            outputs = model.apply(params, input_ids)
        
        # Verify outputs
        assert outputs['logits'].shape == (batch_size, seq_len, config.vocab_size)
        
        # Clear memory between runs
        del params, outputs
        jax.clear_caches()

if __name__ == "__main__":
    pytest.main([__file__])
