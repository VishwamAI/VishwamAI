import pytest
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig

def test_minimal_tokenizer():
    # Basic config with minimum required parameters
    config = ConceptualTokenizerConfig(
        vocab_size=64,
        max_length=512,
        model_prefix="test_tokenizer"
    )
    
    # Initialize tokenizer with config
    tokenizer = ConceptualTokenizer(config)
    
    # Test tokenization
    text = "Test equation"
    tokens = tokenizer.encode(text)
    assert isinstance(tokens, list)

    # Test decoding
    decoded = tokenizer.decode(tokens)
    assert isinstance(decoded, str)

if __name__ == "__main__":
    pytest.main(["-v", __file__])
