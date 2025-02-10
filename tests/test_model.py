import pytest
import torch
import torch.nn.functional as F
from vishwamai.model import (
    VishwamaiModel, 
    VishwamaiConfig,
    MoELayer,
    VishwamaiBlock
)

def test_model_initialization():
    config = VishwamaiConfig()
    model = VishwamaiModel(config)
    assert model is not None
    assert isinstance(model, VishwamaiModel)

def test_forward_pass():
    config = VishwamaiConfig()
    model = VishwamaiModel(config)
    model.to("cpu")  # Ensure model is on CPU for testing
    input_ids = torch.randint(0, config.vocab_size, (2, 128))
    logits = model(input_ids)
    assert logits.shape == (2, 128, config.vocab_size), "Output shape mismatch!"

def test_attention():
    config = VishwamaiConfig()
    model = VishwamaiModel(config)
    block = model.blocks[0]  # Get the first transformer block
    input_ids = torch.randint(0, config.vocab_size, (2, 128))
    hidden_states = model.embeddings(input_ids)
    output = block.attention(hidden_states)
    assert output.shape == hidden_states.shape, "Attention output shape mismatch!"

def test_mlp():
    config = VishwamaiConfig()
    model = VishwamaiModel(config)
    block = model.blocks[0]  # Get the first transformer block
    input_tensor = torch.randn(2, 128, config.hidden_size)
    output = block.mlp(input_tensor)
    assert output.shape == input_tensor.shape, "MLP output shape mismatch!"

def test_rms_norm():
    config = VishwamaiConfig()
    model = VishwamaiModel(config)
    norm_layer = model.ln_f
    input_tensor = torch.randn(2, 128, config.hidden_size)
    output = norm_layer(input_tensor)
    assert output.shape == input_tensor.shape, "RMSNorm output shape mismatch!"

def test_device_transfer():
    config = VishwamaiConfig()
    model = VishwamaiModel(config)
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    assert next(model.parameters()).device.type in ["cuda", "cpu"], "Device transfer failed!"

def test_moe_layer():
    config = VishwamaiConfig()
    moe = MoELayer(config)
    input_tensor = torch.randn(2, 128, config.hidden_size)
    output = moe(input_tensor)
    assert output.shape == input_tensor.shape, "MoE output shape mismatch!"
    
def test_expert_routing():
    config = VishwamaiConfig()
    moe = MoELayer(config)
    input_tensor = torch.randn(2, 128, config.hidden_size)
    
    # Test routing computation
    route_scores = moe.gate(input_tensor)
    assert route_scores.shape == (2, 128, config.n_routed_experts), "Routing scores shape mismatch!"
    
    # Test expert selection
    routing_weights = F.softmax(route_scores, dim=-1)
    top_k_weights, top_k_indices = torch.topk(routing_weights, k=config.n_activated_experts, dim=-1)
    assert top_k_indices.shape == (2, 128, config.n_activated_experts), "Expert selection shape mismatch!"

def test_block_with_moe():
    config = VishwamaiConfig()
    block = VishwamaiBlock(config)
    input_tensor = torch.randn(2, 128, config.hidden_size)
    output = block(input_tensor)
    assert output.shape == input_tensor.shape, "Block with MoE output shape mismatch!"

def test_model_memory_efficiency():
    config = VishwamaiConfig(max_batch_size=2, max_seq_len=128)
    model = VishwamaiModel(config)
    
    # Test memory allocation
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        model.to('cuda')
        input_ids = torch.randint(0, config.vocab_size, (2, 128)).cuda()
        _ = model(input_ids)
        max_memory = torch.cuda.max_memory_allocated()
        assert max_memory > 0, "Model should allocate GPU memory"

def test_expert_load_balancing():
    config = VishwamaiConfig()
    moe = MoELayer(config)
    
    # Test with multiple batches to check load balancing
    batches = [torch.randn(2, 128, config.hidden_size) for _ in range(5)]
    expert_counts = torch.zeros(config.n_routed_experts)
    
    for batch in batches:
        route_scores = moe.gate(batch)
        routing_weights = F.softmax(route_scores, dim=-1)
        _, top_k_indices = torch.topk(routing_weights, k=config.n_activated_experts, dim=-1)
        for expert_idx in range(config.n_routed_experts):
            expert_counts[expert_idx] += (top_k_indices == expert_idx).sum().item()
    
    # Check if load is reasonably balanced
    std_dev = torch.std(expert_counts)
    mean = torch.mean(expert_counts)
    coefficient_of_variation = std_dev / mean
    assert coefficient_of_variation < 0.5, "Expert load is not well balanced"

@pytest.mark.parametrize("batch_size", [1, 2, 4])
@pytest.mark.parametrize("seq_length", [64, 128, 256])
def test_model_different_sizes(batch_size, seq_length):
    config = VishwamaiConfig()
    model = VishwamaiModel(config)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    output = model(input_ids)
    assert output.shape == (batch_size, seq_length, config.vocab_size), \
        f"Output shape mismatch for batch_size={batch_size}, seq_length={seq_length}"

if __name__ == "__main__":
    pytest.main([__file__])
