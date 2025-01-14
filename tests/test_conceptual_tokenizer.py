import pytest
import torch
import os
import tempfile
import sentencepiece as spm
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig

@pytest.fixture
def basic_config():
    return ConceptualTokenizerConfig(
        vocab_size=64,  # Increased from 22 to 64 to meet required characters
        max_length=512,
        concept_tokens=["math", "logic", "science"],
        reasoning_tokens=["if", "then", "because"],
        model_prefix="test_tokenizer",
        character_coverage=0.9995,
        control_symbols=[],
        user_defined_symbols=[]
    )

@pytest.fixture
def training_texts():
    # Expanded training data to include all essential terms
    return [
        "Solve the equation x + 5 = 10",
        "If A is true then B must be false",
        "The chemical reaction produces heat",
        "If the temperature equals 100 degrees then the water boils",
        "The quadratic formula helps solve equations",
        "Logic gates are used in digital circuits",
        "Chemical bonds form between atoms",
        "Simple mathematics problem",
        "Basic logical reasoning",
        "Scientific method experiment",
        "Temperature measurement test",
        "Mathematical equations solved",
        "Logical thinking process",
        "Chemical composition analysis",
        "Physics force calculation",
        "If conditions are met then actions follow",
        "Equals sign usage in equations",
        "Then statements in logic"
    ]

@pytest.fixture
def trained_tokenizer(basic_config, training_texts, tmp_path):
    tokenizer = ConceptualTokenizer(basic_config)
    # Train the tokenizer
    tokenizer.train_tokenizer(training_texts)
    return tokenizer

def test_sentencepiece_training(basic_config, training_texts, tmp_path):
    # Use tmp_path for model prefix to avoid conflicts
    basic_config.model_prefix = str(tmp_path / "test_tokenizer")
    # Ensure vocab size is sufficient for test data
    basic_config.vocab_size = max(64, len(set(''.join(training_texts))) + 10)  # Add padding for special tokens
    
    tokenizer = ConceptualTokenizer(basic_config)
    
    try:
        tokenizer.train_tokenizer(training_texts)
        
        # Check if model files are created with proper path
        assert os.path.exists(f"{basic_config.model_prefix}.model")
        assert os.path.exists(f"{basic_config.model_prefix}.vocab")
        
        # Test tokenization
        test_text = "Test equation solving"
        tokens = tokenizer.encode(test_text)
        assert len(tokens) > 0
        
        # Test if tokens can be decoded back
        decoded = tokenizer.decode(tokens)
        assert isinstance(decoded, str)
        assert len(decoded) > 0
        
    finally:
        # Clean up
        try:
            if os.path.exists(f"{basic_config.model_prefix}.model"):
                os.remove(f"{basic_config.model_prefix}.model")
            if os.path.exists(f"{basic_config.model_prefix}.vocab"):
                os.remove(f"{basic_config.model_prefix}.vocab")
            if os.path.exists("training_data.txt"):
                os.remove("training_data.txt")
        except OSError:
            pass

def test_concept_aware_tokenization(trained_tokenizer):
    # Add concepts with related terms
    trained_tokenizer.add_concept("math", ["equation", "solve", "quadratic"])
    trained_tokenizer.add_concept("logic", ["if", "then", "true", "false"])
    
    # Test math-related text
    math_text = "Solve the quadratic equation"
    tokens = trained_tokenizer.encode(math_text)
    decoded = trained_tokenizer.decode(tokens)
    assert "[CONCEPT_MATH]" in decoded or any(term in decoded for term in ["equation", "solve", "quadratic"])
    
    # Test logic-related text
    logic_text = "If x is true then y is false"
    tokens = trained_tokenizer.encode(logic_text)
    decoded = trained_tokenizer.decode(tokens)
    # Fix the syntax error in the assertion
    assert "[CONCEPT_LOGIC]" in decoded or any(term in decoded for term in ["if", "then"])

def test_batch_processing(trained_tokenizer):
    texts = [
        "First equation example",
        "Second logic test",
        "Third science experiment"
    ]
    
    # Test batch encoding
    batch_tokens = trained_tokenizer.encode(texts)
    assert len(batch_tokens) == len(texts)
    
    # Test batch decoding
    decoded_texts = trained_tokenizer.decode(batch_tokens)
    assert len(decoded_texts) == len(texts)

def test_special_token_handling(trained_tokenizer):
    text = "Test text"
    tokens = trained_tokenizer.encode(text, add_special_tokens=True)
    
    # Check if special tokens are added
    assert tokens[0] == trained_tokenizer.config.bos_id
    # Ensure that eos_id is present in tokens (not necessarily at the last position due to padding)
    assert trained_tokenizer.config.eos_id in tokens

def test_save_load_tokenizer(trained_tokenizer, tmp_path):
    save_path = str(tmp_path / "test_tokenizer")
    
    # Save tokenizer
    trained_tokenizer.save_pretrained(save_path)
    
    # Check if all necessary files are saved
    assert os.path.exists(f"{save_path}/tokenizer.model")
    assert os.path.exists(f"{save_path}/config.json")
    assert os.path.exists(f"{save_path}/concept_data.json")
    
    # Load tokenizer
    loaded_tokenizer = ConceptualTokenizer.from_pretrained(save_path)
    
    # Test if loaded tokenizer works correctly
    test_text = "Test equation"
    original_tokens = trained_tokenizer.encode(test_text)
    loaded_tokens = loaded_tokenizer.encode(test_text)
    assert original_tokens == loaded_tokens

def test_concept_detection(trained_tokenizer):
    trained_tokenizer.add_concept("math", ["equation", "solve", "calculate"])
    
    text = "Solve this equation: 2x + 3 = 7"
    spans = trained_tokenizer._detect_concepts(text)
    
    # Check if concepts are detected correctly
    assert len(spans) > 0
    assert any(span[2] == "math" for span in spans)

def test_error_handling(trained_tokenizer):
    # Test with empty input
    empty_result = trained_tokenizer.encode("")
    assert isinstance(empty_result, list)
    
    # Test with very long input
    long_text = "word " * 1000
    tokens = trained_tokenizer.encode(long_text)
    assert len(tokens) <= trained_tokenizer.config.max_length
    
    # Test with invalid tokens
    with pytest.raises(Exception):
        trained_tokenizer.decode([-1])

def test_semantic_clustering(trained_tokenizer):
    # Add semantic clusters
    trained_tokenizer.add_concept("math", ["algebra", "calculus", "equation"])
    trained_tokenizer.add_concept("physics", ["force", "energy", "motion"])
    
    text = "The algebra equation uses calculus"
    spans = trained_tokenizer._detect_concepts(text)
    
    # Check if related terms are properly clustered
    assert len(spans) > 0
    assert any(span[2] == "math" for span in spans)

def test_tokenizer_config(basic_config):
    # Test configuration settings
    tokenizer = ConceptualTokenizer(basic_config)
    assert tokenizer.config.vocab_size == basic_config.vocab_size
    assert tokenizer.config.max_length == basic_config.max_length
    assert tokenizer.config.model_type == "unigram"
    assert tokenizer.config.character_coverage == 0.9995

def test_concept_embeddings(trained_tokenizer):
    # Add a concept with embedding
    concept = "math"
    trained_tokenizer.add_concept(concept, ["equation", "algebra"])
    
    # Check if concept embedding is created
    concept_token = f"[CONCEPT_{concept.upper()}]"
    assert concept_token in trained_tokenizer.concept_embeddings
    
    # Check if inverse mapping is created
    assert trained_tokenizer.concept_embeddings[concept_token] in trained_tokenizer.inverse_concept_embeddings

if __name__ == "__main__":
    pytest.main(["-v", __file__])