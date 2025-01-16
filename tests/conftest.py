import pytest
from vishwamai.model import VishwamaiModel, VishwamaiConfig
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig
from vishwamai.generate import VishwamaiGenerator, GenerationConfig

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
def model(config):
    model = VishwamaiModel(config)
    model.eval()
    return model

@pytest.fixture
def tokenizer():
    config = ConceptualTokenizerConfig(vocab_size=100, max_length=32)
    tokenizer = ConceptualTokenizer(config)
    return tokenizer

@pytest.fixture
def generator(model, tokenizer):
    config = GenerationConfig(
        max_length=10,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        num_return_sequences=1
    )
    generator = VishwamaiGenerator(model, tokenizer, config)
    return generator