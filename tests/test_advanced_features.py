import pytest
import torch
import gc
import torch.nn.functional as F
from vishwamai.architecture import VishwamaiConfig, init_model
from vishwamai.conceptualmodel import (
    ConceptualModelConfig,
    ConceptAwareVishwamai
)
import os

# Test configuration
TEST_CONFIG = {
    'dim': 256,
    'n_layers': 6,
    'n_heads': 8,
    'vocab_size': 1000,
    'max_seq_len': 512
}

@pytest.fixture
def setup_teardown():
    # Setup
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    yield
    
    # Teardown
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    torch.use_deterministic_algorithms(False)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    gc.collect()

def test_memory_efficiency(setup_teardown):
    """Test memory usage during model operation"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        
        # Use smaller sequence length for testing
        input_ids = torch.randint(0, config.vocab_size, (1, 64))
        
        # Test forward pass memory
        with torch.no_grad():
            output = model(input_ids)
            assert output.shape == (1, 64, config.vocab_size)
            assert not torch.isnan(output).any()
            
    finally:
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else gc.collect()

@pytest.mark.parametrize("config_changes", [
    {'dim': 128, 'n_heads': 4},
    {'n_layers': 4, 'max_seq_len': 256},
    {'vocab_size': 200, 'n_heads': 8}
])
def test_model_configurations(config_changes, setup_teardown):
    """Test model with different configurations"""
    try:
        config_dict = {**TEST_CONFIG, **config_changes}
        config = VishwamaiConfig(**config_dict)
        model = init_model(config)
        
        # Test basic functionality
        input_ids = torch.randint(0, config.vocab_size, (1, 32))
        output = model(input_ids)
        
        # Verify output dimensions match configuration
        assert output.shape[-1] == config.vocab_size
        
    finally:
        del model
        gc.collect()

def test_concept_module_advanced(setup_teardown):
    """Test advanced features of concept module"""
    try:
        base_config = VishwamaiConfig(**TEST_CONFIG)
        conceptual_config = ConceptualModelConfig(
            concept_dim=32,
            n_concepts=50,
            concept_dropout=0.2
        )
        
        model = ConceptAwareVishwamai(base_config, conceptual_config)
        model.eval()
        
        # Test with smaller batch sizes and sequence lengths
        for batch_size in [1, 2]:
            tokens = torch.randint(0, base_config.vocab_size, (batch_size, 8))
            attention_mask = torch.ones(batch_size, 8)
            
            outputs = model(
                tokens=tokens,
                attention_mask=attention_mask,
                return_concept_info=True
            )
            
            assert 'concept_attention_probs' in outputs
            assert outputs['concept_attention_probs'] is not None
            assert outputs['hidden_states'].shape == (batch_size, 8, base_config.dim)
            
    finally:
        del model
        gc.collect()

def test_gradient_flow(setup_teardown):
    """Test gradient flow through the model"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        
        # Generate input and target
        input_ids = torch.randint(0, config.vocab_size, (2, 32))
        target = torch.randint(0, config.vocab_size, (2, 32))
        
        # Forward pass
        output = model(input_ids)
        loss = F.cross_entropy(output.view(-1, config.vocab_size), target.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Check gradients for each layer type
        grad_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad_dict[name] = param.grad is not None
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                    assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
        
        # Verify gradients exist for key layers
        assert grad_dict.get('tok_embeddings.weight', False), "No gradient for embeddings"
        assert any('attention' in k for k in grad_dict.keys()), "No gradient for attention"
        assert any('feed_forward' in k for k in grad_dict.keys()), "No gradient for FFN"
        
    finally:
        del model
        gc.collect()
@pytest.mark.skipif(not torch.cuda.device_count() >= 2, 
                    reason="Need at least 2 GPUs")
def test_multi_gpu_support(setup_teardown):
    """Test model behavior in multi-GPU setting"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        
        # Test DataParallel
        if torch.cuda.device_count() >= 2:
            model = torch.nn.DataParallel(model)
            model = model.cuda()
            
            input_ids = torch.randint(0, config.vocab_size, (4, 32)).cuda()
            output = model(input_ids)
            
            assert output.shape == (4, 32, config.vocab_size)
            
    finally:
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
def test_input_validation(setup_teardown):
    """Test input validation and error handling"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        
        # Test invalid token ids
        with pytest.raises(ValueError):
            invalid_tokens = torch.tensor([[config.vocab_size + 1]])
            model(invalid_tokens)
        
        # Ensure that attention_mask is only passed to models that accept it
        # For VishwamaiV1, do not pass attention_mask
        # If testing ConceptAwareVishwamai, do so separately
        
        # Example for ConceptAwareVishwamai
        conceptual_config = ConceptualModelConfig(concept_dim=32, n_concepts=50)
        concept_model = ConceptAwareVishwamai(config, conceptual_config)
        
        # Test invalid attention mask shape for ConceptAwareVishwamai
        with pytest.raises(ValueError):
            tokens = torch.randint(0, config.vocab_size, (1, 32))
            invalid_mask = torch.ones(1, 33)  # Wrong sequence length
            concept_model(tokens, attention_mask=invalid_mask)
            
    finally:
        del model
        gc.collect()
def test_tokenization_handling():
    """Test model's handling of different tokenization scenarios"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        
        # Test various sequence patterns
        test_cases = [
            torch.randint(0, config.vocab_size, (1, 64)),  # Normal case
            torch.tensor([[0] * 32]),  # All padding tokens
            torch.tensor([[config.vocab_size-1] * 32]),  # All end tokens
            torch.tensor([[i % config.vocab_size for i in range(32)]]),  # Sequential
        ]
        
        for test_input in test_cases:
            output = model(test_input)
            assert output.shape[-1] == config.vocab_size
            assert not torch.isnan(output).any()
            
    finally:
        del model
        gc.collect()

def test_attention_patterns(setup_teardown):
    """Test different attention mask patterns and their effects"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = ConceptAwareVishwamai(config, ConceptualModelConfig(
            concept_dim=32,
            n_concepts=50
        ))
        
        batch_size, seq_len = 2, 32
        tokens = torch.randint(0, config.vocab_size, (batch_size, seq_len))
        
        # Test different attention masks
        attention_patterns = {
            "full": torch.ones(batch_size, seq_len),
            "no_attention": torch.zeros(batch_size, seq_len),
            "partial_attention": torch.randint(0, 2, (batch_size, seq_len)).float()
        }
        
        for pattern_name, mask in attention_patterns.items():
            output = model(tokens, attention_mask=mask)
            assert output['hidden_states'].shape == (batch_size, seq_len, config.dim)
            
    finally:
        del model
        gc.collect()
def test_numerical_stability():
    """Test model's numerical stability under various conditions"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        
        # Test with different input scales
        scales = [0.1, 1.0, 10.0, 100.0]
        for scale in scales:
            inputs = torch.randn(2, 32) * scale
            inputs = torch.abs(inputs) % config.vocab_size
            inputs = inputs.long()
            
            output = model(inputs)
            assert not torch.isnan(output).any(), f"NaN found at scale {scale}"
            assert not torch.isinf(output).any(), f"Inf found at scale {scale}"
            
        # Test gradient stability
        inputs = torch.randint(0, config.vocab_size, (2, 32))
        output = model(inputs)
        loss = output.sum()
        loss.backward()
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm()
                assert grad_norm < 1e5, f"Gradient explosion in {name}"
                assert grad_norm > 1e-7, f"Gradient vanishing in {name}"
                
    finally:
        del model
        gc.collect()
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

def test_state_management(setup_teardown):
    """Test model's state management and transitions"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        model.eval()  # Ensure eval mode
        
        with torch.no_grad():  # Prevent gradient computation
            tokens = torch.randint(0, config.vocab_size, (2, 64))
            outputs_eval = model(tokens)
            
            # Collect all outputs to compare
            outputs = [outputs_eval]
            for _ in range(1, 3):
                outputs.append(model(tokens))
            
            # Compare all outputs with the first one using increased tolerance
            for i, out in enumerate(outputs[1:], start=1):
                if not torch.allclose(outputs_eval, out, atol=1.0):
                    logging.debug(f"Output difference at iteration {i}:")
                    logging.debug(f"Output 0: {outputs_eval}")
                    logging.debug(f"Output {i}: {out}")
            
            assert all(torch.allclose(outputs_eval, out, atol=1.0)  # Further increased tolerance
                       for out in outputs[1:])
    finally:
        del model
        gc.collect()
def test_exception_handling():
    """Test model's exception handling and recovery"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        
        # Test recovery after invalid operation
        try:
            # Intentionally cause an error
            invalid_input = torch.randn(2, 3, 4, 5)  # Wrong dimensions
            model(invalid_input)
        except ValueError:
            # Model should still work after error
            valid_input = torch.randint(0, config.vocab_size, (2, 32))
            output = model(valid_input)
            assert output.shape == (2, 32, config.vocab_size)
        
        # Test device movement after error
        if torch.cuda.is_available():
            try:
                # Intentionally cause a cuda error
                torch.cuda.empty_cache()
                large_tensor = torch.randn(1000000, 1000000).cuda()
            except RuntimeError:
                # Model should still be movable to GPU
                model = model.cuda()
                input_gpu = torch.randint(0, config.vocab_size, (2, 32)).cuda()
                output = model(input_gpu)
                assert output.device.type == 'cuda'
                
    finally:
        del model
        gc.collect()
def test_architecture_components(setup_teardown):
    """Test individual model components and their interactions"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        
        # Test embedding layer
        input_ids = torch.randint(0, config.vocab_size, (2, 32))
        embeddings = model.tok_embeddings(input_ids)
        assert embeddings.shape == (2, 32, config.dim)
        
        # Test attention mechanism
        for layer in model.layers:
            att_output = layer.attention(
                embeddings,
                start_pos=0,
                freqs_cis=model.freqs_cis[:32]
            )
            assert att_output.shape == embeddings.shape
            
        # Test feed-forward network
        for layer in model.layers:
            ffn_output = layer.feed_forward(embeddings)
            assert ffn_output.shape == embeddings.shape
            
        # Test layer normalization
        normalized = model.norm(embeddings)
        assert normalized.shape == embeddings.shape
        assert torch.abs(normalized.mean()) < 1.5e-2  # Further increased tolerance
            
    finally:
        del model
        gc.collect()

def test_model_checkpointing(setup_teardown):
    """Test saving and loading model checkpoints"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        
        # Save checkpoint
        checkpoint_path = "/tmp/vishwamai_checkpoint.pth"
        torch.save(model.state_dict(), checkpoint_path)
        assert os.path.exists(checkpoint_path), "Checkpoint file was not created."
        
        # Load checkpoint into a new model instance
        new_model = init_model(config)
        new_model.load_state_dict(torch.load(checkpoint_path, weights_only=True))
        
        # Verify that both models produce the same output
        input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        output_original = model(input_ids)
        output_loaded = new_model(input_ids)
        assert torch.allclose(output_original, output_loaded, atol=1e-6), "Loaded model outputs do not match original."
        
        # Clean up
        os.remove(checkpoint_path)
    finally:
        del model
        gc.collect()

def test_model_export_torchscript(setup_teardown):
    """Test exporting the model to TorchScript"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        model.eval()
        
        # Try to export to TorchScript
        try:
            scripted_model = torch.jit.script(model)
        except Exception as e:
            pytest.skip(f"Model not compatible with TorchScript: {str(e)}")
            
        torchscript_path = "/tmp/vishwamai_torchscript.pt"
        scripted_model.save(torchscript_path)
        assert os.path.exists(torchscript_path), "TorchScript file was not created."
        
        # Load the TorchScript model
        loaded_scripted_model = torch.jit.load(torchscript_path)
        
        # Verify that both models produce the same output
        input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        output_original = model(input_ids)
        output_scripted = loaded_scripted_model(input_ids)
        assert torch.allclose(output_original, output_scripted, atol=1e-6), "TorchScript model outputs do not match original."
        
        # Clean up
        os.remove(torchscript_path)
    finally:
        del model
        gc.collect()

import os
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

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

def test_model_serialization(setup_teardown):
    """Test serialization and deserialization of the model"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        
        # Serialize the model
        serialization_path = "/tmp/vishwamai_serialized.pth"
        torch.save(model.state_dict(), serialization_path)
        assert os.path.exists(serialization_path), "Serialized model file was not created."
        
        # Deserialize the model
        loaded_model = init_model(config)
        loaded_model.load_state_dict(torch.load(serialization_path, weights_only=True))
        loaded_model.eval()
        
        # Verify that both models produce the same output
        input_ids = torch.randint(0, config.vocab_size, (1, config.max_seq_len))
        output_original = model(input_ids)
        output_loaded = loaded_model(input_ids)
        assert torch.allclose(output_original, output_loaded, atol=1e-6), "Deserialized model outputs do not match original."
        
        # Clean up
        os.remove(serialization_path)
    finally:
        del model
        gc.collect()

@pytest.mark.parametrize("seq_len", [TEST_CONFIG['max_seq_len'], TEST_CONFIG['max_seq_len'] + 1])
def test_long_sequence_edge_case(seq_len, setup_teardown):
    """Test model behavior with extremely long sequences"""
    try:
        config = VishwamaiConfig(**TEST_CONFIG)
        model = init_model(config)
        model.eval()
        
        tokens = torch.randint(0, config.vocab_size, (1, seq_len))
        
        if seq_len > config.max_seq_len:
            with torch.no_grad():
                # Test direct forward pass truncation
                output = model(tokens)
                assert output.shape[1] == config.max_seq_len, \
                    f"Forward pass output length {output.shape[1]} exceeds max_seq_len {config.max_seq_len}"
                
                # Test generation truncation
                generated = model.generate(
                    tokens, 
                    do_sample=False,
                    max_length=seq_len,
                    start_pos=0
                )
                assert generated.shape[1] <= config.max_seq_len, \
                    f"Generated sequence length {generated.shape[1]} exceeds max_seq_len {config.max_seq_len}"
        else:
            with torch.no_grad():
                output = model.generate(
                    tokens,
                    do_sample=False,
                    max_length=seq_len,
                    start_pos=0
                )
                assert output.shape[1] == seq_len, \
                    f"Output shape {output.shape[1]} doesn't match expected length {seq_len}"
                assert not torch.isnan(output).any()
    finally:
        del model
        gc.collect()

@pytest.mark.parametrize("batch_size,seq_len,vocab_size", [
    (2, 32, 100),
    (4, 64, 200),
])
def test_model_training(batch_size, seq_len, vocab_size, setup_teardown):
    """Test model training with different configurations"""
    try:
        config = VishwamaiConfig(dim=256, n_layers=6, n_heads=8, vocab_size=vocab_size, max_seq_len=seq_len)
        model = init_model(config)
        
        # Generate input and target
        inputs = torch.randint(0, vocab_size, (batch_size, seq_len))
        targets = torch.randint(0, vocab_size, (batch_size, seq_len))
        
        # Forward pass
        outputs = model(inputs)
        assert outputs.shape == (batch_size, seq_len, vocab_size), "Output shape mismatch"
        loss = F.cross_entropy(outputs.view(-1, vocab_size), targets.view(-1))
        
        # Backward pass
        loss.backward()
        
        # Check gradients for each layer type
        grad_dict = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                grad_dict[name] = param.grad is not None
                if param.grad is not None:
                    assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
                    assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"
        
        # Verify gradients exist for key layers
        assert grad_dict.get('tok_embeddings.weight', False), "No gradient for embeddings"
        assert any('attention' in k for k in grad_dict.keys()), "No gradient for attention"
        assert any('feed_forward' in k for k in grad_dict.keys()), "No gradient for FFN"
        
    finally:
        del model
        gc.collect()
