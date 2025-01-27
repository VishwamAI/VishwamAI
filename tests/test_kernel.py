import pytest
import torch
import numpy as np
from vishwamai.kernel import act_quant, weight_dequant, fp8_gemm

# Check if CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()
try:
    import triton
    TRITON_AVAILABLE = True
except ImportError:
    TRITON_AVAILABLE = False

# CPU implementation of quantization for testing when CUDA/Triton not available
def cpu_quantize(x: torch.Tensor, block_size: int = 128):
    """CPU fallback implementation of quantization"""
    assert x.is_contiguous()
    x_reshaped = x.view(-1, block_size)
    
    # Handle zero and near-zero inputs
    abs_max = torch.max(torch.abs(x_reshaped), dim=1)[0]
    scale = torch.where(
        abs_max > 0,
        abs_max / 448.0,
        torch.ones_like(abs_max)  # Use scale=1.0 for all-zero blocks
    )
    
    # Safe division with handling for zero scale
    y = torch.where(
        scale.unsqueeze(1) > 0,
        x_reshaped / scale.unsqueeze(1),
        torch.zeros_like(x_reshaped)
    )
    
    y = y.view_as(x)
    return y.to(torch.float32), scale  # Use float32 instead of float8 for CPU

def cpu_dequantize(x: torch.Tensor, scale: torch.Tensor):
    """CPU fallback implementation of dequantization"""
    if x.dim() == 1:
        return x * scale
    return x * scale.view(*scale.shape, 1)

@pytest.fixture
def setup_tensors():
    """Fixture to create test tensors"""
    torch.manual_seed(42)  # For reproducibility
    device = torch.device('cuda' if CUDA_AVAILABLE else 'cpu')
    
    # Create sample tensors
    x = torch.randn(128, 128, device=device)  # Smaller matrices for CPU testing
    w = torch.randn(128, 128, device=device)
    s = torch.ones(8, 8, device=device)
    
    return x, w, s

def get_cuda_arch():
    """Get CUDA architecture version if available"""
    if not CUDA_AVAILABLE:
        return 0
    device = torch.cuda.current_device()
    capabilities = torch.cuda.get_device_capability(device)
    return capabilities[0] * 10 + capabilities[1]

CUDA_ARCH = get_cuda_arch()
SUPPORTS_FP8 = CUDA_ARCH >= 89

@pytest.mark.skipif(not TRITON_AVAILABLE or not SUPPORTS_FP8, reason="Triton not available or FP8 not supported")
def test_act_quant_gpu(setup_tensors):
    """Test activation quantization on GPU"""
    x, _, _ = setup_tensors
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")
        
    y, s = act_quant(x, block_size=128)
    
    # Basic checks
    assert y.shape == x.shape, "Output shape should match input shape"
    assert s.shape == (x.size(0), x.size(1) // 128), "Scale shape should match block configuration"
    assert torch.all(torch.abs(y) <= 448.0), "FP8 values should be within bounds"

def test_act_quant_cpu(setup_tensors):
    """Test activation quantization on CPU"""
    x, _, _ = setup_tensors
    x = x.cpu()
    
    y, s = cpu_quantize(x, block_size=128)
    
    # Basic checks
    assert y.shape == x.shape, "Output shape should match input shape"
    assert s.shape == (x.size(0),), "Scale shape should be correct"
    assert torch.all(torch.isfinite(y)), "All values should be finite"

@pytest.mark.skipif(not TRITON_AVAILABLE or not SUPPORTS_FP8, reason="Triton not available or FP8 not supported")
def test_weight_dequant_gpu(setup_tensors):
    """Test weight dequantization on GPU"""
    _, w, s = setup_tensors
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")
        
    # First quantize weights
    y, s_actual = act_quant(w, block_size=128)
    
    # Test dequantization
    w_dequant = weight_dequant(y, s_actual)
    
    assert w_dequant.shape == w.shape, "Output shape should match input shape"
    relative_error = torch.abs((w_dequant - w) / (w + 1e-7))
    assert torch.mean(relative_error) < 0.1, "Average relative error should be reasonable"

def test_weight_dequant_cpu(setup_tensors):
    """Test weight dequantization on CPU"""
    _, w, _ = setup_tensors
    w = w.cpu()
    
    # Use CPU quantization
    y, s = cpu_quantize(w, block_size=128)
    w_dequant = cpu_dequantize(y, s)
    
    assert w_dequant.shape == w.shape, "Output shape should match input shape"
    assert torch.all(torch.isfinite(w_dequant)), "All values should be finite"

def test_edge_cases_cpu():
    """Test edge cases on CPU"""
    # Test small tensor
    x_small = torch.randn(64, 64)
    y_small, s_small = cpu_quantize(x_small, block_size=32)
    assert y_small.shape == x_small.shape, "Should handle small tensors"
    
    # Test zero tensor
    x_zero = torch.zeros(128, 128)
    y_zero, s_zero = cpu_quantize(x_zero)
    assert torch.all(y_zero == 0), "Should handle zero tensor correctly"
    assert torch.all(torch.isfinite(y_zero)), "Should not produce NaN values"
    assert torch.all(s_zero == 1.0), "Scale should be 1.0 for zero tensors"
    
    # Test maximum values
    x_max = torch.ones(128, 128) * 1e6
    y_max, s_max = cpu_quantize(x_max)
    assert torch.all(torch.isfinite(y_max)), "Should handle large values without NaN/inf"
    
    # Test mixed zero and non-zero blocks
    x_mixed = torch.zeros(128, 128)
    x_mixed[64:, :] = 1.0  # Half zeros, half ones
    y_mixed, s_mixed = cpu_quantize(x_mixed)
    assert torch.all(torch.isfinite(y_mixed)), "Should handle mixed zero/non-zero blocks"
    assert torch.all(y_mixed[:64] == 0), "Zero blocks should remain zero"
    assert torch.any(y_mixed[64:] != 0), "Non-zero blocks should be preserved"
    
    # Test near-zero values
    x_tiny = torch.ones(128, 128) * 1e-10
    y_tiny, s_tiny = cpu_quantize(x_tiny)
    assert torch.all(torch.isfinite(y_tiny)), "Should handle near-zero values"

def test_numerical_stability_cpu():
    """Test numerical stability with various input patterns on CPU"""
    # Test with very small values
    x_small_val = torch.randn(128, 128) * 1e-6
    y_small, s_small = cpu_quantize(x_small_val)
    w_dequant = cpu_dequantize(y_small, s_small)
    assert torch.all(torch.isfinite(w_dequant)), "Should handle very small values"
    
    # Test with mixed scales
    x_mixed = torch.cat([
        torch.randn(64, 128),
        torch.randn(64, 128) * 1e-6
    ], dim=0)
    y_mixed, s_mixed = cpu_quantize(x_mixed)
    assert torch.all(torch.isfinite(y_mixed)), "Should handle mixed scales"

@pytest.mark.skipif(not TRITON_AVAILABLE or not SUPPORTS_FP8, reason="Triton not available or FP8 not supported")
def test_gpu_cpu_consistency(setup_tensors):
    """Test consistency between GPU and CPU implementations"""
    if not CUDA_AVAILABLE:
        pytest.skip("CUDA not available")
        
    x, _, _ = setup_tensors
    x_cpu = x.cpu()
    
    # GPU quantization
    y_gpu, s_gpu = act_quant(x)
    y_gpu = y_gpu.cpu()
    s_gpu = s_gpu.cpu()
    
    # CPU quantization
    y_cpu, s_cpu = cpu_quantize(x_cpu)
    
    # Compare results (allowing for some numerical differences)
    assert torch.allclose(torch.abs(y_gpu), torch.abs(y_cpu), rtol=0.1, atol=0.1), \
        "GPU and CPU quantization should be reasonably similar"

if __name__ == "__main__":
    pytest.main([__file__])
