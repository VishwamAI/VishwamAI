# Import required modules
import torch
from transformers import AutoTokenizer
from datasets import load_dataset

# Import VishwamAI components
from vishwamai.base_layers import Linear
from vishwamai.Transformer import Transformer
from vishwamai import (
    create_model,
    ModelArgs,
    VishwamAITokenizer,
    TokenizerConfig
)

def test_imports():
    print("Testing imports...")
    
    # Test ModelArgs
    args = ModelArgs(
        max_batch_size=4,
        max_seq_len=2048,
        dtype="fp8",
        vocab_size=32000,
        dim=1024,
        n_layers=12,
        n_heads=16
    )
    print("✓ ModelArgs imported successfully")
    
    # Test Transformer
    try:
        transformer = Transformer(args)
        print("✓ Transformer imported and instantiated successfully")
    except Exception as e:
        print(f"✗ Error instantiating Transformer: {str(e)}")

    # Test Tokenizer
    try:
        tokenizer = VishwamAITokenizer(TokenizerConfig(
            vocab_size=32000,
            max_sentence_length=2048
        ))
        print("✓ VishwamAITokenizer imported and instantiated successfully")
    except Exception as e:
        print(f"✗ Error instantiating VishwamAITokenizer: {str(e)}")

if __name__ == "__main__":
    test_imports()
