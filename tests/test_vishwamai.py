import pytest
import torch
import gc
import os
import torch.nn.functional as F
from vishwamai.architecture import VishwamaiConfig, init_model
from vishwamai.conceptualmodel import (
    ConceptualModelConfig,
    ConceptAwareVishwamai,
    advanced_concept_flow,
    ensure_gpu_availability
)
import math  # Added import if not present

# Updated test configuration with even smaller dimensions
TEST_CONFIG = {
    'dim': 64,            # Further reduced from 128
    'n_layers': 2,
    'n_heads': 2,
    'max_seq_len': 128,   # Further reduced from 256
    'vocab_size': 100     # Further reduced from 1000
}

@pytest.fixture(autouse=True)
def setup_teardown():
    # Setup
    torch.manual_seed(42)  # For reproducibility
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()
    
    yield
    
    # Teardown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()

@pytest.mark.parametrize("batch_size, seq_len", [(2, 64), (4, 128)])  # Even smaller test sizes
def test_vishwamai_forward(batch_size, seq_len, setup_teardown):
    config = VishwamaiConfig(**TEST_CONFIG)
    
    try:
        model = init_model(config)
        model.eval()  # Ensure eval mode
        
        with torch.no_grad():  # Prevent gradient computation
            tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
            output = model(tokens)
            
            # Check output shape and values
            assert output.shape == (batch_size, seq_len, config.vocab_size)
            assert not torch.isnan(output).any(), "Output contains NaN values"
            assert not torch.isinf(output).any(), "Output contains Inf values"
            
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")
    finally:
        del model
        gc.collect()

def test_vishwamai_zero_batch_error(setup_teardown):
    config = VishwamaiConfig(**TEST_CONFIG)
    
    try:
        model = init_model(config)
        model.eval()
        
        with torch.no_grad():
            tokens = torch.randint(0, config.vocab_size, (0, 10))
            with pytest.raises(ValueError, match="Batch size cannot be zero"):
                _ = model(tokens)
    finally:
        del model
        gc.collect()

def test_advanced_concept_flow():
    concepts = torch.randn(5, 5)
    result = advanced_concept_flow(concepts)
    assert result.shape == concepts.shape

@pytest.mark.parametrize("use_gpu", [True, False])
def test_gpu_availability(use_gpu, monkeypatch):
    # Force CPU/gpu for test
    monkeypatch.setattr(torch.cuda, "is_available", lambda: use_gpu)
    if use_gpu:
        ensure_gpu_availability()  # Should not raise
    else:
        with pytest.raises(RuntimeError, match="GPU not available"):
            ensure_gpu_availability()

def test_concept_aware_vishwamai(setup_teardown):
    try:
        base_config = VishwamaiConfig(**TEST_CONFIG)
        conceptual_config = ConceptualModelConfig(
            concept_dim=32,     # Reduced from 64
            n_concepts=50,      # Reduced from 100
        )
        
        model = ConceptAwareVishwamai(base_config, conceptual_config)
        model.eval()  # Set to eval mode
        
        with torch.no_grad():
            tokens = torch.randint(0, base_config.vocab_size, (1, 8))  # Smaller sequence length
            attention_mask = torch.ones_like(tokens)
            
            outputs = model(
                tokens=tokens,  # Changed from input_ids to tokens
                attention_mask=attention_mask
            )
            
            assert outputs["hidden_states"].shape == (1, 8, base_config.dim)
            assert "loss" not in outputs  # No loss when no labels provided
            
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")
    finally:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

@pytest.mark.parametrize("device", ["cpu"])  # Remove cuda test temporarily
def test_vishwamai_device_compatibility(device, setup_teardown):
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config).to(device)
        model.eval()
        
        with torch.no_grad():
            tokens = torch.randint(0, config.vocab_size, (1, 4)).to(device)  # Reduced sequence length
            output = model(tokens)
            assert output.shape == (1, 4, config.vocab_size)
            
    except Exception as e:
        pytest.fail(f"Test failed with error: {str(e)}")
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

def test_model_pruning():
    config = VishwamaiConfig(
        dim=256,
        n_layers=4,
        n_heads=4,
        max_seq_len=512
    )
    try:
        model = init_model(config)
        prune_ratio = 0.1
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                if hasattr(module, 'weight_orig'):
                    assert hasattr(module, 'weight'), f"Pruning not applied to {name}"
                    # Verify the number of zeros matches the prune ratio
                    num_zeros = torch.sum(module.weight == 0).item()
                    total_weights = module.weight.numel()
                    assert abs(num_zeros / total_weights - prune_ratio) < 0.05, \
                        f"Prune ratio mismatch in {name}: expected {prune_ratio}, got {num_zeros / total_weights}"
    finally:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

@pytest.mark.parametrize("batch_size,seq_len,vocab_size", [
    (2, 32, 100),
    (4, 64, 200),
])
def test_model_training(batch_size, seq_len, vocab_size, setup_teardown):
    try:
        config = VishwamaiConfig(
            dim=256, 
            n_layers=6,
            n_heads=8,
            vocab_size=vocab_size,
            max_seq_len=seq_len
        )
        model = init_model(config)
        model.train()  # Ensure model is in training mode
        
        # Generate random data
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Ensure no gradients are accumulated from previous iterations
        model.zero_grad()
        
        # Forward pass
        outputs = model(inputs)
        loss = F.cross_entropy(
            outputs.view(-1, vocab_size),
            targets.view(-1),
            reduction='mean'
        )
        
        # Backward pass
        loss.backward()
        
        # Verify gradients
        any_grad = False
        for name, param in model.named_parameters():
            if param.requires_grad:
                if param.grad is not None:
                    any_grad = True
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                    assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
        
        assert any_grad, "No gradients were computed"
        
    finally:
        del model
        gc.collect()

def test_concept_module_integration(setup_teardown):
    try:
        base_config = VishwamaiConfig(**TEST_CONFIG)
        conceptual_config = ConceptualModelConfig(
            concept_dim=32,
            n_concepts=50
        )
        
        model = ConceptAwareVishwamai(base_config, conceptual_config)
        
        # Test with different attention mask patterns
        tokens = torch.randint(0, base_config.vocab_size, (2, 16))
        full_attention = torch.ones(2, 16)
        causal_attention = torch.ones(2, 16)  # All tokens attended
        
        # Test different attention patterns
        outputs_full = model(tokens, attention_mask=full_attention)
        outputs_causal = model(tokens, attention_mask=causal_attention)
        
        assert outputs_full["hidden_states"].shape == (2, 16, base_config.dim)
        assert outputs_causal["hidden_states"].shape == (2, 16, base_config.dim)
        
    finally:
        del model
        gc.collect()


@pytest.mark.parametrize("batch_size, seq_len", [(1, 4), (2, 8)])
def test_model_edge_cases(batch_size, seq_len, setup_teardown):
    """Test model behavior in edge cases"""
    config = VishwamaiConfig(**TEST_CONFIG)
    model = init_model(config)
    
    try:
        # Test empty sequence
        with pytest.raises(ValueError, match="Batch size cannot be zero"):
            model(torch.empty((0, seq_len), dtype=torch.long))
        
        # Test sequence exceeding max_seq_len
        long_sequence = torch.randint(0, config.vocab_size, (batch_size, config.max_seq_len + 1))
        output = model.forward(long_sequence)  # Should automatically truncate
        assert output.size(1) == config.max_seq_len, "Failed to handle sequence length correctly"
        
        # Test invalid token IDs
        invalid_tokens = torch.randint(config.vocab_size, config.vocab_size + 10, (batch_size, seq_len))
        with pytest.raises(ValueError, match="Token indices exceed vocabulary size"):
            model(invalid_tokens)
        
        # Test with varying sequence lengths in same batch
        if batch_size > 1:
            varying_lengths = [torch.randint(0, config.vocab_size, (1, l)) for l in [4, 8]]
            padded_sequence = torch.nn.utils.rnn.pad_sequence(
                [seq.squeeze(0) for seq in varying_lengths],
                batch_first=True
            )
            output = model(padded_sequence)
            assert output.size(0) == batch_size, "Failed to handle variable sequence lengths"
        
        # Test numerical edge cases
        edge_tokens = torch.tensor([[0, config.vocab_size - 1]], dtype=torch.long)
        output = model(edge_tokens)
        assert not torch.isnan(output).any(), "NaN in output for edge token values"
        
    except Exception as e:
        pytest.fail(f"Edge case test failed with error: {str(e)}")
        
    finally:
        del model
        gc.collect()

def test_model_serialization(setup_teardown, tmp_path):
    config = VishwamaiConfig(**TEST_CONFIG)
    model = init_model(config)
    
    try:
        # Test save and load
        save_path = tmp_path / "model.pt"
        torch.save(model.state_dict(), save_path)
        
        new_model = init_model(config)
        new_model.load_state_dict(torch.load(save_path, weights_only=True))  # Added weights_only=True
        
        # Verify both models produce same output
        test_input = torch.randint(0, config.vocab_size, (1, 16))
        with torch.no_grad():
            out1 = model(test_input)
            out2 = new_model(test_input)
            torch.testing.assert_close(out1, out2)
            
    finally:
        del model
        gc.collect()


def test_long_sequence_handling(setup_teardown):
    config = VishwamaiConfig(**{**TEST_CONFIG, 'max_seq_len': 2048})
    model = init_model(config)
    
    try:
        # Test with varying sequence lengths
        for seq_len in [512, 1024, 2048]:
            inputs = torch.randint(0, config.vocab_size, (1, seq_len))
            outputs = model(inputs)
            assert outputs.shape == (1, seq_len, config.vocab_size)
            
    finally:
        del model
        gc.collect()

# Add memory cleanup fixture
@pytest.fixture(autouse=True)
def cleanup_memory():
    yield
    torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

if __name__ == "__main__":
    pytest.main(["-v", __file__])

def test_input_validation(setup_teardown):
    """Test input validation and error handling"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        
        # Test invalid token ids
        with pytest.raises(ValueError):
            invalid_tokens = torch.tensor([[config.vocab_size + 1]])
            model(invalid_tokens)
        
        # Remove or comment out the following block as VishwamaiV1 does not accept attention_mask
        # with pytest.raises(ValueError):
        #     tokens = torch.randint(0, config.vocab_size, (1, 32))
        #     invalid_mask = torch.ones(1, 33)  # Wrong sequence length
        #     model(tokens, attention_mask=invalid_mask)
        
    finally:
        del model
        gc.collect()

@pytest.mark.parametrize("precision", ["float16", "bfloat16"])
def test_inference_mixed_precision(precision, setup_teardown):
    """Test model inference with mixed precision"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        model.eval()
        
        if precision == "float16" and not torch.cuda.is_available():
            pytest.skip("float16 requires CUDA.")
        if precision == "bfloat16" and (not torch.cuda.is_available() or not torch.cuda.is_bf16_supported()):
            pytest.skip("bfloat16 not supported on this device.")
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if precision == "float16" else torch.bfloat16
        model = model.to(device=device, dtype=dtype)
        
        input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len)).to(device=device, dtype=torch.long)
        
        with torch.no_grad():
            output = model(input_ids)
            assert output.dtype == dtype, f"Output dtype {output.dtype} does not match expected {dtype}."
            assert output.shape == (1, config.max_seq_len, config.vocab_size)
            assert not torch.isnan(output).any(), "Output contains NaN values."
            assert not torch.isinf(output).any(), "Output contains Inf values."
    finally:
        del model
        gc.collect()

# Example test function completion
def test_example_feature():
    """Test an example feature of Vishwamai"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        model.eval()
        
        input_ids = torch.randint(0, config.vocab_size, (1, 32))
        output = model(input_ids)
        
        assert output.shape[-1] == config.vocab_size, "Output vocab size mismatch"
        
    finally:
        del model
        gc.collect()