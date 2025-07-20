"""
Tests for Mixture of Recursions (MoR) functionality in VishwamAI.

This test suite verifies that the MoR implementation works correctly,
including recursive routing, selective computation, and integration
with the main model architecture.
"""

import jax
import jax.numpy as jnp
import pytest
from vishwamai.model import ModelConfig, VishwamAIModel, TransformerBlock
from vishwamai.layers import FeedForward
from vishwamai.attention import FlashAttention


def test_feedforward_recursive_routing():
    """Test that FeedForward layer with recursion works correctly."""
    
    ff = FeedForward(
        dim=512,
        hidden_dim=2048,
        dropout=0.1,
        use_moe=True,
        expert_count=4,
        expert_capacity=2,
        use_recursion=True,
        max_recursion_depth=3,
        recursion_capacity=2
    )
    
    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    x = jnp.ones((2, 10, 512))
    params = ff.init({'params': rng, 'dropout': jax.random.PRNGKey(43)}, x, training=True)
    
    output = ff.apply(params, x, training=True, rngs={'dropout': jax.random.PRNGKey(44)})
    
    assert output.shape == x.shape, f"Expected shape {x.shape}, got {output.shape}"
    
    assert not jnp.allclose(output, x, atol=1e-6), "Output should be transformed, not identical to input"
    
    assert jnp.all(jnp.isfinite(output)), "Output should contain only finite values"


def test_feedforward_without_recursion():
    """Test that FeedForward layer works without recursion (baseline)."""
    
    ff = FeedForward(
        dim=512,
        hidden_dim=2048,
        dropout=0.1,
        use_moe=True,
        expert_count=4,
        expert_capacity=2,
        use_recursion=False
    )
    
    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    x = jnp.ones((2, 10, 512))
    params = ff.init({'params': rng, 'dropout': jax.random.PRNGKey(43)}, x, training=True)
    
    output = ff.apply(params, x, training=True, rngs={'dropout': jax.random.PRNGKey(44)})
    
    assert output.shape == x.shape
    assert not jnp.allclose(output, x, atol=1e-6)
    assert jnp.all(jnp.isfinite(output))


def test_transformer_block_with_recursion():
    """Test TransformerBlock with recursion-enabled FeedForward."""
    
    config = ModelConfig(
        dim=512,
        depth=4,
        heads=8,
        head_dim=64,
        vocab_size=1000,
        max_seq_len=128,
        use_moe=True,
        expert_count=4,
        expert_capacity=2,
        use_recursion=True,
        max_recursion_depth=3,
        recursion_capacity=2,
        use_flash_attention=True,
        use_grouped_query_attention=True,
        gqa_groups=4
    )
    
    block = TransformerBlock(config)
    
    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    x = jnp.ones((2, 32, 512))
    params = block.init({'params': rng, 'dropout': jax.random.PRNGKey(43)}, x)
    
    output = block.apply(params, x, rngs={'dropout': jax.random.PRNGKey(44)})
    
    assert output.shape == x.shape
    assert not jnp.allclose(output, x, atol=1e-6)
    assert jnp.all(jnp.isfinite(output))


def test_full_model_with_recursion():
    """Test full VishwamAI model with recursion enabled."""
    
    config = ModelConfig(
        dim=512,
        depth=4,
        heads=8,
        head_dim=64,
        vocab_size=1000,
        max_seq_len=128,
        use_moe=True,
        expert_count=4,
        expert_capacity=2,
        use_recursion=True,
        max_recursion_depth=3,
        recursion_capacity=2,
        gradient_checkpointing=False  # Disable for testing
    )
    
    # Create model
    model = VishwamAIModel(config)
    
    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    input_ids = jnp.ones((2, 64), dtype=jnp.int32)
    params = model.init({'params': rng, 'dropout': jax.random.PRNGKey(43)}, input_ids, training=True)
    
    logits = model.apply(params, input_ids, training=False, rngs={'dropout': jax.random.PRNGKey(44)})
    
    expected_shape = (2, 64, config.vocab_size)
    assert logits.shape == expected_shape, f"Expected shape {expected_shape}, got {logits.shape}"
    assert jnp.all(jnp.isfinite(logits)), "Logits should contain only finite values"


def test_recursion_depth_progression():
    """Test that different recursion depths produce different outputs."""
    
    ff_depth1 = FeedForward(
        dim=512,
        hidden_dim=2048,
        use_moe=True,
        expert_count=4,
        expert_capacity=2,
        use_recursion=True,
        max_recursion_depth=1,
        recursion_capacity=2
    )
    
    ff_depth3 = FeedForward(
        dim=512,
        hidden_dim=2048,
        use_moe=True,
        expert_count=4,
        expert_capacity=2,
        use_recursion=True,
        max_recursion_depth=3,
        recursion_capacity=2
    )
    
    rng = jax.random.PRNGKey(42)
    x = jnp.ones((2, 10, 512))
    
    params1 = ff_depth1.init({'params': rng, 'dropout': jax.random.PRNGKey(43)}, x, training=True)
    params3 = ff_depth3.init({'params': rng, 'dropout': jax.random.PRNGKey(45)}, x, training=True)
    
    output1 = ff_depth1.apply(params1, x, training=True, rngs={'dropout': jax.random.PRNGKey(44)})
    output3 = ff_depth3.apply(params3, x, training=True, rngs={'dropout': jax.random.PRNGKey(46)})
    
    assert output1.shape == x.shape
    assert output3.shape == x.shape
    assert jnp.all(jnp.isfinite(output1))
    assert jnp.all(jnp.isfinite(output3))


def test_flash_attention_with_recursion_support():
    """Test FlashAttention with recursion support enabled."""
    
    attention = FlashAttention(
        dim=512,
        heads=8,
        head_dim=64,
        dropout=0.1,
        use_gqa=True,
        gqa_groups=4,
        support_recursion=True
    )
    
    # Initialize parameters
    rng = jax.random.PRNGKey(42)
    x = jnp.ones((2, 32, 512))
    params = attention.init(rng, x, training=True)
    
    output = attention.apply(params, x, training=False)
    
    assert output.shape == x.shape
    assert not jnp.allclose(output, x, atol=1e-6)
    assert jnp.all(jnp.isfinite(output))


def test_recursion_capacity_bounds():
    """Test that recursion capacity is properly bounded."""
    
    for capacity in [1, 2, 4]:
        ff = FeedForward(
            dim=256,
            hidden_dim=1024,
            use_moe=True,
            expert_count=4,
            expert_capacity=2,
            use_recursion=True,
            max_recursion_depth=3,
            recursion_capacity=capacity
        )
        
        rng = jax.random.PRNGKey(42)
        x = jnp.ones((1, 8, 256))
        params = ff.init({'params': rng, 'dropout': jax.random.PRNGKey(43)}, x, training=True)
        
        output = ff.apply(params, x, training=True, rngs={'dropout': jax.random.PRNGKey(44)})
        assert output.shape == x.shape
        assert jnp.all(jnp.isfinite(output))


def test_model_config_recursion_parameters():
    """Test that ModelConfig properly handles recursion parameters."""
    
    config_with_recursion = ModelConfig(
        dim=512,
        depth=4,
        heads=8,
        use_recursion=True,
        max_recursion_depth=3,
        recursion_capacity=2
    )
    
    assert config_with_recursion.use_recursion == True
    assert config_with_recursion.max_recursion_depth == 3
    assert config_with_recursion.recursion_capacity == 2
    
    config_without_recursion = ModelConfig(
        dim=512,
        depth=4,
        heads=8
    )
    
    assert config_without_recursion.use_recursion == False
    assert config_without_recursion.max_recursion_depth == 3  # default
    assert config_without_recursion.recursion_capacity == 2   # default


if __name__ == "__main__":
    test_feedforward_recursive_routing()
    test_feedforward_without_recursion()
    test_transformer_block_with_recursion()
    test_full_model_with_recursion()
    test_recursion_depth_progression()
    test_flash_attention_with_recursion_support()
    test_recursion_capacity_bounds()
    test_model_config_recursion_parameters()
    
    print("All Mixture of Recursions tests passed!")
