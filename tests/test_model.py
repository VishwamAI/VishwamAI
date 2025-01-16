import pytest
import torch
from vishwamai.model import VishwamaiModel, VishwamaiConfig

@pytest.fixture
def config():
    return VishwamaiConfig(
        vocab_size=100,
        hidden_size=64,
        num_hidden_layers=2,
        num_attention_heads=2,
        intermediate_size=128,
        max_position_embeddings=32
    )

def test_model_init(config):
    model = VishwamaiModel(config)
    assert isinstance(model, VishwamaiModel)
    
def test_model_forward(config):
    model = VishwamaiModel(config)
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    
    outputs = model(input_ids)
    assert outputs.shape == (batch_size, seq_length, config.vocab_size)
    
def test_model_attention_mask(config):
    model = VishwamaiModel(config)
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
    attention_mask = torch.ones((batch_size, seq_length))
    attention_mask[:, seq_length//2:] = 0
    
    outputs = model(input_ids, attention_mask=attention_mask)
    assert outputs.shape == (batch_size, seq_length, config.vocab_size)
