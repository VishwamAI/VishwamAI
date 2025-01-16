import pytest
import torch
from vishwamai.generate import VishwamaiGenerator, GenerationConfig
from vishwamai.model import VishwamaiModel, VishwamaiConfig

class MockTokenizer:
    def __init__(self):
        pass  # remove the direct assignment
    
    @property
    def eos_token_id(self):
        return 2
        
    def encode(self, text, return_tensors=None):
        # Mock encoding that returns tensor of indices
        input_ids = torch.tensor([[1, 3, 4, 5]])
        if return_tensors == "pt":
            return input_ids
        return input_ids.tolist()
        
    def batch_decode(self, ids, skip_special_tokens=True):
        # Mock decoding that returns list of strings
        return ["Generated text"]

@pytest.fixture
def mock_model():
    config = VishwamaiConfig(vocab_size=100)
    model = VishwamaiModel(config)
    model.eval()
    return model

def test_generator_init(mock_model):
    generator = VishwamaiGenerator(
        model=mock_model,
        tokenizer=MockTokenizer(),
        config=GenerationConfig()
    )
    assert isinstance(generator, VishwamaiGenerator)

def test_generate_text(mock_model):
    generator = VishwamaiGenerator(
        model=mock_model,
        tokenizer=MockTokenizer(),
        config=GenerationConfig(max_length=10)
    )
    
    output = generator.generate("Test input")
    assert isinstance(output, list)
    assert len(output) > 0
    assert isinstance(output[0], str)
