import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pytest
import torch
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig
from vishwamai.model import VishwamaiModel, VishwamaiConfig
from vishwamai.generate import VishwamaiGenerator, GenerationConfig

@pytest.fixture
def config():
    return VishwamaiConfig()

@pytest.fixture
def model(config):
    return VishwamaiModel(config)

@pytest.fixture
def tokenizer():
    config = ConceptualTokenizerConfig(
        vocab_size=100,  # Small vocab for testing
        pad_token="<pad>",
        bos_token="<s>",
        eos_token="</s>",
        pad_id=0,
        bos_id=1,
        eos_id=2,
        unk_id=3,
        model_type="unigram"
    )
    
    tokenizer = ConceptualTokenizer(config)
    
    # Train tokenizer on test data
    with open("tests/test_data.txt", "r") as f:
        texts = f.readlines()
    tokenizer.train_tokenizer(texts)
    
    # Add test concepts after training
    tokenizer.add_concept("TEST", ["test", "testing", "tested"])
    tokenizer.add_concept("INPUT", ["input", "inputs", "inputted"])
    return tokenizer

@pytest.fixture
def generator(model, tokenizer):
    gen_config = GenerationConfig()
    return VishwamaiGenerator(model, tokenizer, gen_config)
