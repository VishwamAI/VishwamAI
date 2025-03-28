"""Test cases for VishwamAI kernels across different platforms."""

import os
import pytest
import numpy as np
import jax
import jax.numpy as jnp
import torch

from vishwamai.kernels.core.kernel import (
    KernelConfig,
    multi_head_attention_kernel,
    flash_attention,
    rope_embedding
)
from vishwamai.kernels.tpu import (
    gemm,
    layer_norm,
    flash_attention as tpu_flash_attn
)
from vishwamai.kernels.cuda import flash_mla_with_kvcache
from vishwamai.kernels.ops.sparse import SparseMatrixOps
from vishwamai.kernels.ops.hybrid_matmul import HybridMatMul
from vishwamai.kernels.optimizers.moe_balance import rebalance_experts

# Test configurations
TEST_SIZES = [(128, 128), (512, 512), (1024, 1024)]
TEST_DTYPES = [np.float32, np.float16]
PLATFORMS = ['cpu', 'gpu', 'tpu']

@pytest.fixture
def test_data():
    """Generate test data."""
    np.random.seed(42)
    return {
        'small': np.random.randn(128, 128).astype(np.float32),
        'medium': np.random.randn(512, 512).astype(np.float32),
        'large': np.random.randn(1024, 1024).astype(np.float32)
    }

@pytest.fixture
def attention_data():
    """Generate attention test data."""
    batch_size, num_heads, seq_len, head_dim = 2, 8, 256, 64
    return {
        'query': np.random.randn(batch_size, num_heads, seq_len, head_dim),
        'key': np.random.randn(batch_size, num_heads, seq_len, head_dim),
        'value': np.random.randn(batch_size, num_heads, seq_len, head_dim),
        'mask': np.random.randint(0, 2, (batch_size, num_heads, seq_len, seq_len))
    }

class TestMatrixOperations:
    """Test matrix operation kernels."""
    
    @pytest.mark.parametrize("size", TEST_SIZES)
    @pytest.mark.parametrize("dtype", TEST_DTYPES)
    def test_matmul(self, test_data, size, dtype):
        """Test matrix multiplication."""
        data = test_data['small'].astype(dtype)
        expected = np.matmul(data, data.T)
        
        # Test TPU GEMM
        if jax.devices('tpu'):
            tpu_result = gemm(
                jnp.array(data),
                jnp.array(data.T),
                block_size=128
            )
            np.testing.assert_allclose(tpu_result, expected, rtol=1e-3)
        
        # Test GPU GEMM
        if torch.cuda.is_available():
            gpu_data = torch.tensor(data, device='cuda')
            gpu_result = torch.matmul(gpu_data, gpu_data.T).cpu().numpy()
            np.testing.assert_allclose(gpu_result, expected, rtol=1e-3)
    
    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_hybrid_matmul(self, test_data, size):
        """Test hybrid matrix multiplication."""
        data = test_data['small']
        hybrid_mm = HybridMatMul(block_size=128)
        result = hybrid_mm(data, data.T)
        expected = np.matmul(data, data.T)
        np.testing.assert_allclose(result, expected, rtol=1e-3)

class TestAttentionMechanisms:
    """Test attention operation kernels."""
    
    def test_flash_attention(self, attention_data):
        """Test flash attention implementation."""
        q, k, v = [attention_data[key] for key in ['query', 'key', 'value']]
        mask = attention_data['mask']
        
        # Test TPU flash attention
        if jax.devices('tpu'):
            tpu_result = tpu_flash_attn(
                jnp.array(q),
                jnp.array(k),
                jnp.array(v),
                mask=jnp.array(mask)
            )
            assert tpu_result.shape == q.shape
        
        # Test GPU flash attention
        if torch.cuda.is_available():
            gpu_result = flash_mla_with_kvcache(
                torch.tensor(q, device='cuda'),
                torch.tensor(k, device='cuda'),
                torch.tensor(v, device='cuda')
            )
            assert gpu_result.shape == q.shape
    
    def test_rope_embedding(self, test_data):
        """Test rotary position embedding."""
        data = test_data['small']
        result = rope_embedding(jnp.array(data), dim=data.shape[-1])
        assert result.shape == data.shape

class TestSparseOperations:
    """Test sparse operation kernels."""
    
    def test_sparse_matmul(self, test_data):
        """Test sparse matrix multiplication."""
        dense = test_data['small']
        sparse_ops = SparseMatrixOps()
        
        # Create sparse data
        indices = np.random.randint(0, dense.shape[0], (100, 2))
        values = np.random.randn(100)
        
        result = sparse_ops.block_sparse_matmul(
            jnp.array(dense),
            jnp.array(values),
            jnp.array(indices)
        )
        assert result.shape == dense.shape

class TestOptimizers:
    """Test optimizer kernels."""
    
    def test_expert_balancing(self):
        """Test MoE load balancing."""
        num_experts = 8
        batch_size = 32
        hidden_dim = 64
        
        # Create random expert weights
        expert_weights = np.random.randn(batch_size, num_experts)
        inputs = np.random.randn(batch_size, hidden_dim)
        
        balanced = rebalance_experts(
            jnp.array(expert_weights),
            jnp.array(inputs)
        )
        assert balanced.shape == expert_weights.shape

class TestLayerOperations:
    """Test layer operation kernels."""
    
    @pytest.mark.parametrize("size", TEST_SIZES)
    def test_layer_norm(self, test_data, size):
        """Test layer normalization."""
        data = test_data['small']
        scale = np.ones(data.shape[-1])
        bias = np.zeros(data.shape[-1])
        
        # Test TPU layer norm
        if jax.devices('tpu'):
            tpu_result = layer_norm(
                jnp.array(data),
                jnp.array(scale),
                jnp.array(bias)
            )
            assert tpu_result.shape == data.shape
        
        # Compare with PyTorch implementation
        if torch.cuda.is_available():
            torch_ln = torch.nn.LayerNorm(data.shape[-1])
            torch_result = torch_ln(torch.tensor(data)).detach().numpy()
            np.testing.assert_allclose(tpu_result, torch_result, rtol=1e-3)

@pytest.mark.integration
class TestEndToEnd:
    """End-to-end integration tests."""
    
    def test_full_attention_block(self, attention_data):
        """Test complete attention block with all components."""
        q, k, v = [attention_data[key] for key in ['query', 'key', 'value']]
        mask = attention_data['mask']
        
        # Test attention with layer norm and residual
        norm_q = layer_norm(jnp.array(q))
        
        attn_output = flash_attention(
            norm_q,
            jnp.array(k),
            jnp.array(v),
            mask=jnp.array(mask)
        )
        
        assert attn_output.shape == q.shape
    
    def test_hybrid_operations(self, test_data):
        """Test hybrid dense-sparse operations."""
        data = test_data['medium']
        
        # Create sparse component
        sparse_ops = SparseMatrixOps()
        indices = np.random.randint(0, data.shape[0], (1000, 2))
        values = np.random.randn(1000)
        
        # Test hybrid computation
        hybrid_mm = HybridMatMul(block_size=128)
        sparse_result = sparse_ops.block_sparse_matmul(
            jnp.array(data),
            jnp.array(values),
            jnp.array(indices)
        )
        dense_result = hybrid_mm(data, data.T)
        
        final_result = jnp.add(sparse_result, dense_result)
        assert final_result.shape == data.shape

if __name__ == '__main__':
    pytest.main(['-v', __file__])
