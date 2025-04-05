"""Test suite for VishwamAI model analysis."""

import pytest
import jax
import jax.numpy as jnp
from vishwamai.model import VishwamAI, VishwamAIConfig
from vishwamai.layers import TPUMultiHeadAttention

@pytest.fixture
def model_config():
    """Create a test model configuration."""
    return VishwamAIConfig(
        vocab_size=32000,
        hidden_dim=512,  # Reduced for testing
        num_layers=2,    # Reduced for testing
        num_heads=8,
        head_dim=32,
        mlp_dim=1024,
        max_seq_len=128,
        dropout_rate=0.1,
        attention_dropout=0.1,
        use_flash_attn=False  # Disabled for CPU testing
    )

@pytest.fixture
def model(model_config):
    """Create a test model instance."""
    return VishwamAI(config=model_config)

def test_model_initialization(model, model_config, dummy_input, rng_key):
    """Test model initialization and parameter shapes."""
    # Initialize parameters
    variables = model.init(rng_key, dummy_input)
    
    # Check parameter tree structure
    assert 'params' in variables
    params = variables['params']
    
    # Validate embedding layer
    assert 'token_embedding' in params
    assert params['token_embedding']['embedding'].shape == \
           (model_config.vocab_size, model_config.hidden_dim)
    
    # Validate transformer blocks
    for i in range(model_config.num_layers):
        block_name = f'transformer_block_{i}'
        assert block_name in params
        block_params = params[block_name]
        
        # Check attention parameters
        assert 'attention' in block_params
        assert 'qkv' in block_params['attention']
        qkv_weight = block_params['attention']['qkv']['kernel']
        expected_qkv_shape = (
            model_config.hidden_dim,
            3 * model_config.num_heads * model_config.head_dim
        )
        assert qkv_weight.shape == expected_qkv_shape
        
        # Check MLP parameters
        assert 'mlp' in block_params
        assert 'fc1' in block_params['mlp']
        mlp_weight = block_params['mlp']['fc1']['kernel']
        assert mlp_weight.shape == (model_config.hidden_dim, model_config.mlp_dim)

def test_forward_pass(model, dummy_input, rng_key):
    """Test model forward pass and output shapes."""
    # Initialize model
    variables = model.init(rng_key, dummy_input)
    
    # Run forward pass
    output = model.apply(variables, dummy_input)
    
    # Check output shape
    batch_size, seq_len = dummy_input.shape
    expected_shape = (batch_size, seq_len, model.config.vocab_size)
    assert output.shape == expected_shape
    assert output.dtype == jnp.float32

def test_attention_mechanism(model_config, rng_key):
    """Test attention computation."""
    batch_size = 2
    seq_len = 32  # Reduced for CPU testing
    
    # Create attention layer
    attention = TPUMultiHeadAttention(
        num_heads=model_config.num_heads,
        head_dim=model_config.head_dim,
        dropout_rate=0.1
    )
    
    # Create inputs
    shape = (batch_size, seq_len, model_config.hidden_dim)
    inputs_q = jnp.ones(shape)
    inputs_kv = jnp.ones(shape)
    
    # Initialize attention
    variables = attention.init(rng_key, inputs_q, inputs_kv)
    
    # Run attention
    output = attention.apply(variables, inputs_q, inputs_kv)
    
    # Verify output shape
    expected_shape = (
        batch_size,
        seq_len,
        model_config.num_heads * model_config.head_dim
    )
    assert output.shape == expected_shape

def test_gradient_flow(model, dummy_input, rng_key):
    """Test gradient computation and backpropagation."""
    # Initialize model
    dummy_labels = jnp.ones_like(dummy_input)
    variables = model.init(rng_key, dummy_input)
    
    # Define loss function
    def compute_loss(params):
        logits = model.apply({'params': params}, dummy_input)
        return jax.nn.softmax_cross_entropy_with_integer_labels(
            logits,
            dummy_labels
        ).mean()
    
    # Compute gradients
    grads = jax.grad(compute_loss)(variables['params'])
    
    # Check gradient shapes match parameter shapes
    jax.tree_util.tree_map(
        lambda g, p: assert_equal_shape(g, p),
        grads,
        variables['params']
    )

def assert_equal_shape(a, b):
    """Assert that two arrays have the same shape."""
    assert a.shape == b.shape, f"Shape mismatch: {a.shape} != {b.shape}"