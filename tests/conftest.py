import pytest
import torch
import gc
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig

@pytest.fixture
def tokenizer_config():
    return ConceptualTokenizerConfig(
        vocab_size=1000,
        concept_tokens=["math", "logic", "science"],
        reasoning_tokens=["if", "then", "because"],
        special_tokens_map={
            "[MATH]": 4,
            "[LOGIC]": 5,
            "[SCIENCE]": 6
        }
    )

@pytest.fixture
def tokenizer(tokenizer_config):
    return ConceptualTokenizer(tokenizer_config)

@pytest.fixture
def concept_embeddings():
    return {
        "math": torch.randn(768),
        "logic": torch.randn(768),
        "science": torch.randn(768)
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