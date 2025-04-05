"""Test suite for VishwamAI model analysis."""

import pytest
import jax
import jax.numpy as jnp
from vishwamai.model import VishwamAI, VishwamAIConfig
from vishwamai.layers import TPUMultiHeadAttention

@pytest.fixture
def model_config():
    """Provide minimal model configuration for testing."""
    return VishwamAIConfig(
        vocab_size=1000,
        hidden_dim=128,
        num_layers=2,
        num_heads=4,
        head_dim=32,
        mlp_dim=512,
        max_seq_len=64,
        dropout_rate=0.1,
        attention_dropout=0.1,
        use_flash_attn=False,
        # Include required generation settings
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        max_branches=2,
        max_depth=2,
        beam_width=2
    )

@pytest.fixture
def model(model_config):
    """Create a minimal model for testing."""
    return VishwamAI(config=model_config)

@pytest.fixture
def dummy_input():
    """Create small dummy input for testing."""
    return jnp.ones((1, 32), dtype=jnp.int32)

@pytest.fixture
def rng_key():
    """Provide random key for testing."""
    return jax.random.PRNGKey(0)

def test_model_initialization(model, dummy_input, rng_key):
    """Test model initialization and parameter shapes."""
    variables = model.init(rng_key, dummy_input)
    assert "params" in variables

def test_forward_pass(model, dummy_input, rng_key):
    """Test model forward pass and output shapes."""
    variables = model.init(rng_key, dummy_input)
    outputs = model.apply(variables, dummy_input)
    assert "logits" in outputs
    assert outputs["logits"].shape[0] == dummy_input.shape[0]

def test_attention_mechanism(model_config, rng_key):
    """Test attention computation."""
    batch_size = 1
    seq_len = 16
    
    attention = TPUMultiHeadAttention(
        num_heads=model_config.num_heads,
        head_dim=model_config.head_dim,
        dropout_rate=0.1
    )
    
    shape = (batch_size, seq_len, model_config.hidden_dim)
    inputs_q = jnp.ones(shape)
    inputs_kv = jnp.ones(shape)
    
    variables = attention.init(rng_key, inputs_q, inputs_kv)
    output = attention.apply(variables, inputs_q, inputs_kv)
    assert output.shape == inputs_q.shape

def test_gradient_flow(model, dummy_input, rng_key):
    """Test gradient computation and backpropagation."""
    variables = model.init(rng_key, dummy_input)
    
    def loss_fn(params):
        outputs = model.apply({"params": params}, dummy_input)
        return jnp.mean(outputs["logits"])
    
    grad_fn = jax.grad(loss_fn)
    grads = grad_fn(variables["params"])
    assert grads is not None