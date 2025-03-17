import unittest
import jax
import jax.numpy as jnp
from vishwamai.transformer import TransformerModel, EnhancedTransformerModel
from vishwamai.layers import MLABlock

class TestTransformer(unittest.TestCase):
    def setUp(self):
        self.config = {
            'num_layers': 2,
            'hidden_size': 256,
            'num_heads': 4,
            'vocab_size': 1000,
            'max_seq_length': 128,
            'dropout_rate': 0.1,
            'dtype': jnp.float32
        }
        self.batch_size = 2
        self.seq_length = 16
        self.rng = jax.random.PRNGKey(0)

    def test_transformer_output_shape(self):
        model = TransformerModel(**self.config)
        params = model.init(self.rng, jnp.ones((1, self.seq_length), dtype=jnp.int32))
        input_ids = jax.random.randint(self.rng, (self.batch_size, self.seq_length), 0, self.config['vocab_size'])
        output = model.apply(params, input_ids)
        expected_shape = (self.batch_size, self.seq_length, self.config['vocab_size'])
        self.assertEqual(output.shape, expected_shape)

    def test_enhanced_transformer(self):
        model = EnhancedTransformerModel(**self.config)
        params = model.init(self.rng, jnp.ones((1, self.seq_length), dtype=jnp.int32))
        input_ids = jax.random.randint(self.rng, (self.batch_size, self.seq_length), 0, self.config['vocab_size'])
        output = model.apply(params, input_ids)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.config['vocab_size']))

if __name__ == '__main__':
    unittest.main()