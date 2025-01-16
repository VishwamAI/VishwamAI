import pytest
import torch
from vishwamai.fp8_cast_bf16 import fp8_cast, weight_dequant

def test_fp8_cast_basic():
    # Test basic casting
    x = torch.randn(10, 10)
    y = fp8_cast(x, torch.bfloat16)
    assert y.dtype == torch.bfloat16
    
def test_fp8_cast_scaling():
    # Test scaling behavior
    x = torch.tensor([1000.0, -1000.0])
    y = fp8_cast(x, torch.bfloat16)
    assert not torch.isinf(y).any()
    assert not torch.isnan(y).any()
    
def test_weight_dequant():
    weight = torch.ones(4, 4)
    scale_inv = torch.tensor(0.5)
    result = weight_dequant(weight, scale_inv)
    assert result.dtype == torch.bfloat16
    assert torch.allclose(result.float(), torch.ones(4, 4) * 0.5, rtol=1e-3)
