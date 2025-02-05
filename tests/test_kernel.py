import pytest
import torch
from vishwamai.kernel import act_quant, weight_dequant, fp8_gemm

@pytest.fixture
def input_tensor():
    if torch.cuda.is_available():
        return torch.randn(128, 128, dtype=torch.float32, device="cuda")
    return torch.randn(128, 128, dtype=torch.float32, device="cpu")

@pytest.fixture
def scale_tensor():
    if torch.cuda.is_available():
        return torch.randn(128, 128, dtype=torch.float32, device="cuda")
    return torch.randn(128, 128, dtype=torch.float32, device="cpu")

def is_cuda_available():
    return torch.cuda.is_available()

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_act_quant(input_tensor, device="cuda" if torch.cuda.is_available() else "cpu"):
    quantized, scales = act_quant(input_tensor)
    assert quantized.dtype == torch.float16, "Quantized tensor dtype mismatch"
    assert scales.dtype == torch.float32, "Scales tensor dtype mismatch"
    assert quantized.shape == input_tensor.shape, "Quantized tensor shape mismatch"
    assert scales.shape == (input_tensor.size(0),), "Scales tensor shape mismatch"

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_weight_dequant(input_tensor, scale_tensor, device="cuda" if torch.cuda.is_available() else "cpu"):
    dequantized = weight_dequant(input_tensor, scale_tensor)
    assert dequantized.dtype == torch.float32, "Dequantized tensor dtype mismatch"
    assert dequantized.shape == input_tensor.shape, "Dequantized tensor shape mismatch"

@pytest.mark.skipif(not is_cuda_available(), reason="CUDA not available")
def test_fp8_gemm(input_tensor, scale_tensor, device="cuda" if torch.cuda.is_available() else "cpu"):
    b = torch.randn(128, 128, dtype=torch.float32, device=device)
    b_s = torch.randn(128, 128, dtype=torch.float32, device=device)
    result = fp8_gemm(input_tensor, scale_tensor, b, b_s)
    assert result.dtype == torch.float32, "FP8 GEMM result dtype mismatch"
    assert result.shape == (128, 128), "FP8 GEMM result shape mismatch"

