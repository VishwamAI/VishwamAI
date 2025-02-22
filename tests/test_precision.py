"""
Tests for precision handling in Vishwamai models
"""
import pytest
import torch
from typing import Dict, Any

from vishwamai.model import (
    create_model,
    ModelArgs,
    VISHWAMAI_TINY,
    VISHWAMAI_BASE
)
from vishwamai.utils.t4_utils import (
    get_device_capabilities,
    enable_t4_optimizations
)

# Enable optimizations before tests
enable_t4_optimizations()

@pytest.fixture
def small_test_config() -> ModelArgs:
    """Get a small model configuration for testing"""
    return VISHWAMAI_TINY.update(
        max_batch_size=2,
        max_seq_len=32,
        use_flash_attention=True,
        use_mixed_precision=True
    )

@pytest.fixture
def device_info() -> Dict[str, Any]:
    """Get device capabilities"""
    return get_device_capabilities()

def test_model_precision_modes(small_test_config: ModelArgs, device_info: Dict[str, Any]):
    """Test model creation and forward pass with different precision modes"""
    if not device_info["cuda_available"]:
        pytest.skip("CUDA not available")
        
    precisions = ["fp16", "fp32"]
    if device_info["bfloat16"]:
        precisions.append("bf16")
        
    for dtype in precisions:
        # Create model with specific precision
        config = small_test_config.update(dtype=dtype)
        model = create_model(config=config)
        
        # Check model dtype
        param_dtype = next(model.parameters()).dtype
        expected_dtype = {
            "fp16": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16
        }[dtype]
        assert param_dtype == expected_dtype, f"Model dtype {param_dtype} does not match expected {expected_dtype}"
        
        # Test forward pass
        batch_size, seq_len = 2, 16
        inputs = {
            "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len)
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        assert outputs["hidden_states"].dtype == expected_dtype
        assert outputs["hidden_states"].shape == (batch_size, seq_len, config.hidden_size)

def test_mixed_precision_training(device_info: Dict[str, Any]):
    """Test mixed precision training"""
    if not device_info["cuda_available"] or not device_info["amp"]:
        pytest.skip("CUDA or AMP not available")
        
    config = VISHWAMAI_TINY.update(
        dtype="fp16",
        use_mixed_precision=True
    )
    
    model = create_model(config=config)
    model.train()
    
    # Create dummy inputs and labels
    batch_size, seq_len = 2, 16
    inputs = {
        "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len)
    }
    labels = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Forward pass
    outputs = model(**inputs)
    loss = torch.nn.functional.cross_entropy(
        outputs["hidden_states"].view(-1, config.hidden_size),
        labels.view(-1)
    )
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    for param in model.parameters():
        if param.requires_grad:
            assert param.grad is not None
            assert not torch.isnan(param.grad).any()
            assert not torch.isinf(param.grad).any()

def test_precision_memory_usage(device_info: Dict[str, Any]):
    """Test memory usage with different precision modes"""
    if not device_info["cuda_available"]:
        pytest.skip("CUDA not available")
        
    base_config = VISHWAMAI_BASE
    batch_size, seq_len = 4, 512
    
    def measure_memory(dtype: str) -> float:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        config = base_config.update(dtype=dtype)
        model = create_model(config=config)
        
        inputs = {
            "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
            "attention_mask": torch.ones(batch_size, seq_len)
        }
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        return torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    fp32_memory = measure_memory("fp32")
    fp16_memory = measure_memory("fp16")
    
    # FP16 should use roughly half the memory
    assert fp16_memory < 0.75 * fp32_memory
    
    if device_info["bfloat16"]:
        bf16_memory = measure_memory("bf16")
        assert bf16_memory < 0.75 * fp32_memory

def test_precision_numerical_accuracy(small_test_config: ModelArgs):
    """Test numerical accuracy across precision modes"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    # Create reference model in FP32
    ref_config = small_test_config.update(dtype="fp32", use_mixed_precision=False)
    ref_model = create_model(config=ref_config)
    
    # Test inputs
    batch_size, seq_len = 2, 16
    inputs = {
        "input_ids": torch.randint(0, ref_config.vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len)
    }
    
    # Get reference outputs
    with torch.no_grad():
        ref_outputs = ref_model(**inputs)
    ref_hidden = ref_outputs["hidden_states"].float()
    
    # Test FP16
    fp16_config = small_test_config.update(dtype="fp16")
    fp16_model = create_model(config=fp16_config)
    with torch.no_grad():
        fp16_outputs = fp16_model(**inputs)
    fp16_hidden = fp16_outputs["hidden_states"].float()
    
    # Compare results (allowing some numerical error)
    fp16_diff = (ref_hidden - fp16_hidden).abs().mean()
    assert fp16_diff < 1e-2, f"FP16 difference too high: {fp16_diff}"
    
    # Test BF16 if available
    if get_device_capabilities()["bfloat16"]:
        bf16_config = small_test_config.update(dtype="bf16")
        bf16_model = create_model(config=bf16_config)
        with torch.no_grad():
            bf16_outputs = bf16_model(**inputs)
        bf16_hidden = bf16_outputs["hidden_states"].float()
        
        bf16_diff = (ref_hidden - bf16_hidden).abs().mean()
        assert bf16_diff < 1e-2, f"BF16 difference too high: {bf16_diff}"

def test_gradient_precision(small_test_config: ModelArgs):
    """Test gradient precision handling"""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
        
    config = small_test_config.update(
        dtype="fp16",
        use_mixed_precision=True
    )
    
    model = create_model(config=config)
    model.train()
    
    # Training step
    batch_size, seq_len = 2, 16
    inputs = {
        "input_ids": torch.randint(0, config.vocab_size, (batch_size, seq_len)),
        "attention_mask": torch.ones(batch_size, seq_len)
    }
    
    # Forward pass
    outputs = model(**inputs)
    loss = outputs["hidden_states"].sum()
    
    # Backward pass
    loss.backward()
    
    # Check gradient types
    for name, param in model.named_parameters():
        if param.requires_grad:
            # Weights should stay in model precision
            assert param.dtype == getattr(torch, config.dtype)
            # Gradients should be in FP32 for stability
            assert param.grad.dtype == torch.float32
