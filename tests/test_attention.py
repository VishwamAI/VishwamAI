import unittest
import jax
import jax.numpy as jnp
from vishwamai.flash_attention import FlashAttention
from vishwamai.layers import MultiHeadAttention

class TestAttention(unittest.TestCase):
    def setUp(self):
        self.batch_size = 2
        self.seq_length = 16
        self.num_heads = 4
        self.head_dim = 64
        self.hidden_size = self.num_heads * self.head_dim
        self.rng = jax.random.PRNGKey(0)

    def test_flash_attention(self):
        flash_attn = FlashAttention(num_heads=self.num_heads, head_dim=self.head_dim)
        q = jax.random.normal(self.rng, (self.batch_size, self.seq_length, self.hidden_size))
        k = jax.random.normal(self.rng, (self.batch_size, self.seq_length, self.hidden_size))
        v = jax.random.normal(self.rng, (self.batch_size, self.seq_length, self.hidden_size))
        
        output = flash_attn.apply({'params': {}}, q, k, v)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def test_multi_head_attention(self):
        mha = MultiHeadAttention(num_heads=self.num_heads, head_dim=self.head_dim)
        params = mha.init(self.rng, 
                         jnp.ones((1, self.seq_length, self.hidden_size)),
                         jnp.ones((1, self.seq_length, self.hidden_size)),
                         jnp.ones((1, self.seq_length, self.hidden_size)))
        
        x = jax.random.normal(self.rng, (self.batch_size, self.seq_length, self.hidden_size))
        output = mha.apply(params, x, x, x)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.hidden_size))

    def test_attention_mask(self):
        mha = MultiHeadAttention(num_heads=self.num_heads, head_dim=self.head_dim)
        params = mha.init(self.rng, 
                         jnp.ones((1, self.seq_length, self.hidden_size)),
                         jnp.ones((1, self.seq_length, self.hidden_size)),
                         jnp.ones((1, self.seq_length, self.hidden_size)))
        
        x = jax.random.normal(self.rng, (self.batch_size, self.seq_length, self.hidden_size))
        mask = jnp.tril(jnp.ones((self.seq_length, self.seq_length)))
        output = mha.apply(params, x, x, x, attention_mask=mask)
        self.assertEqual(output.shape, (self.batch_size, self.seq_length, self.hidden_size))

if __name__ == '__main__':
    unittest.main()