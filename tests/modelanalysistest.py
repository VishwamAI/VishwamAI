"""Test suite for analyzing VishwamAI model architecture and behavior."""

import unittest
import jax
import jax.numpy as jnp
from vishwamai.model import VishwamAI, VishwamAIConfig
from vishwamai.layers import (
    TPUMultiHeadAttention,
    TPUGEMMLinear,
    FlashAttention
)

class ModelAnalysisTest(unittest.TestCase):
    """Test cases for analyzing VishwamAI model."""
    
    def setUp(self):
        """Initialize test configuration and model."""
        self.config = VishwamAIConfig(
            vocab_size=32000,
            hidden_dim=1024,  # Smaller for testing
            num_layers=4,     # Fewer layers for testing
            num_heads=8,
            head_dim=64,
            mlp_dim=2048,
            max_seq_len=512,
            dropout_rate=0.1,
            attention_dropout=0.1,
            use_flash_attn=True
        )
        self.model = VishwamAI(config=self.config)
        self.rng = jax.random.PRNGKey(0)
        
    def test_model_initialization(self):
        """Test model initialization and parameter shapes."""
        batch_size = 2
        seq_len = 128
        
        # Create dummy input
        dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        
        # Initialize parameters
        variables = self.model.init(self.rng, dummy_input)
        
        # Check parameter tree structure
        self.assertIn('params', variables)
        params = variables['params']
        
        # Validate embedding layer
        self.assertIn('token_embedding', params)
        self.assertEqual(
            params['token_embedding']['embedding'].shape,
            (self.config.vocab_size, self.config.hidden_dim)
        )
        
        # Validate transformer blocks
        for i in range(self.config.num_layers):
            block_params = params[f'transformer_block_{i}']
            
            # Check attention parameters
            self.assertIn('attention', block_params)
            qkv_weight = block_params['attention']['qkv']['kernel']
            self.assertEqual(
                qkv_weight.shape,
                (self.config.hidden_dim, 3 * self.config.num_heads * self.config.head_dim)
            )
            
            # Check MLP parameters
            self.assertIn('mlp', block_params)
            mlp_weight = block_params['mlp']['fc1']['kernel']
            self.assertEqual(
                mlp_weight.shape,
                (self.config.hidden_dim, self.config.mlp_dim)
            )

    def test_forward_pass(self):
        """Test model forward pass and output shapes."""
        batch_size = 2
        seq_len = 128
        
        # Initialize model
        dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        variables = self.model.init(self.rng, dummy_input)
        
        # Run forward pass
        output = self.model.apply(variables, dummy_input)
        
        # Check output shape
        expected_shape = (batch_size, seq_len, self.config.vocab_size)
        self.assertEqual(output.shape, expected_shape)
        
        # Check output dtype
        self.assertEqual(output.dtype, jnp.float32)
        
    def test_attention_mechanism(self):
        """Test attention computation and flash attention."""
        batch_size = 2
        seq_len = 128
        
        # Create attention layer
        attention = TPUMultiHeadAttention(
            num_heads=self.config.num_heads,
            head_dim=self.config.head_dim,
            dropout_rate=0.1,
            use_flash_attn=True
        )
        
        # Create dummy input
        x = jnp.ones((batch_size, seq_len, self.config.hidden_dim))
        
        # Initialize attention
        variables = attention.init(self.rng, x)
        
        # Run attention forward pass
        output = attention.apply(variables, x)
        
        # Verify output shape
        self.assertEqual(
            output.shape,
            (batch_size, seq_len, self.config.num_heads * self.config.head_dim)
        )
        
    def test_gradient_flow(self):
        """Test gradient computation and backpropagation."""
        batch_size = 2
        seq_len = 128
        
        # Initialize model
        dummy_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        dummy_labels = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        variables = self.model.init(self.rng, dummy_input)
        
        # Define loss function
        def compute_loss(params):
            logits = self.model.apply({'params': params}, dummy_input)
            return jax.nn.softmax_cross_entropy_with_integer_labels(
                logits,
                dummy_labels
            ).mean()
        
        # Compute gradients
        grads = jax.grad(compute_loss)(variables['params'])
        
        # Check gradient shapes match parameter shapes
        jax.tree_util.tree_map(
            lambda g, p: self.assertEqual(g.shape, p.shape),
            grads,
            variables['params']
        )

if __name__ == '__main__':
    unittest.main()