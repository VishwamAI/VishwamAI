import pytest
import torch
from vishwamai.model import VishwamaiModel, VishwamaiConfig, RotaryEmbedding

@pytest.fixture
def config():
    return VishwamaiConfig(
        vocab_size=100,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        intermediate_size=512,
        max_position_embeddings=32
    )

@pytest.fixture
def model(config, monkeypatch):
    model = VishwamaiModel(config)
    model.eval()
    
    # Mock rotary_emb to prevent None returns
    for block in model.blocks:
        block.attention.rotary_emb = RotaryEmbedding(dim=config.hidden_size // config.num_attention_heads, theta=config.rope_theta)
    
    return model

def test_model_forward(model, config):
    input_ids = torch.randint(0, config.vocab_size, (2, 16))  # batch_size=2, seq_length=16
    output = model(input_ids)
    assert output.shape == (2, 16, config.vocab_size), "Output shape mismatch"

def test_model_with_attention_mask(model, config):
    input_ids = torch.randint(0, config.vocab_size, (1, 10))
    attention_mask = torch.ones(1, 10)
    output = model(input_ids, attention_mask)
    assert output is not None, "Model output is None"
    assert output.shape == (1, 10, config.vocab_size), "Output shape mismatch with attention mask"

def test_model_device_transfer(config):
    model = VishwamaiModel(config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    input_ids = torch.randint(0, config.vocab_size, (1, 10)).to(device)
    output = model(input_ids)
    assert output.device == device, "Model output device mismatch"

def test_model_forward_no_attention_mask(model, config):
    input_ids = torch.randint(0, config.vocab_size, (3, 20))
    output = model(input_ids)
    assert output.shape == (3, 20, config.vocab_size), "Output shape mismatch without attention mask"

def test_model_forward_large_input(model, config):
    input_ids = torch.randint(0, config.vocab_size, (1, config.max_position_embeddings))
    output = model(input_ids)
    assert output.shape == (1, config.max_position_embeddings, config.vocab_size), "Output shape mismatch with large input"
