import pytest
import torch
import pyarrow as pa
from pathlib import Path
from datasets import Dataset
from train_math import create_math_dataloaders, math_collate_fn
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig
from vishwamai.model import VishwamaiModel, VishwamaiConfig

def test_math_collate_fn():
    batch = [
        {'question': 'What is 2+2?', 'answer': '4'},
        {'question': 'What is 3*3?', 'answer': '9'}
    ]
    
    config = ConceptualTokenizerConfig(vocab_size=32000, max_length=512)
    tokenizer = ConceptualTokenizer(config)
    output = math_collate_fn(batch, tokenizer)
    assert 'input_ids' in output
    assert 'concept_ids' in output
    assert isinstance(output['input_ids'], torch.Tensor)
    assert isinstance(output['concept_ids'], torch.Tensor)

class MockMathDataset(Dataset):
    def __init__(self):
        data = {
            'question': ['What is 2+2?', 'What is 3*3?'],
            'answer': ['4', '9']
        }
        arrow_table = pa.Table.from_pydict(data)
        super().__init__(arrow_table)

@pytest.mark.integration
def test_create_math_dataloaders(monkeypatch):
    def mock_load_dataset(*args, **kwargs):
        return {'train': MockMathDataset(), 'test': MockMathDataset()}
    
    monkeypatch.setattr('train_math.load_dataset', mock_load_dataset)
    
    train_loader, val_loader = create_math_dataloaders(batch_size=2)
    
    # Test batch from train loader
    batch = next(iter(train_loader))
    assert 'input_ids' in batch
    assert 'concept_ids' in batch
    
    # Test batch from val loader
    batch = next(iter(val_loader))
    assert 'input_ids' in batch
    assert 'concept_ids' in batch

    # Test batch from train loader
    batch = next(iter(train_loader))
    assert 'input_ids' in batch
    assert 'concept_ids' in batch
    
    # Test batch from val loader
    batch = next(iter(val_loader))
    assert 'input_ids' in batch
    assert 'concept_ids' in batch

def test_model_tokenizer_integration():
    config = VishwamaiConfig()
    model = VishwamaiModel(config)
    tokenizer_config = ConceptualTokenizerConfig(
        vocab_size=config.vocab_size,
        max_length=config.max_position_embeddings
    )
    tokenizer = ConceptualTokenizer(tokenizer_config)
    
    input_text = "Solve for x: 2x + 3 = 11"
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor([input_ids])
    
    logits = model(input_ids)
    assert logits.shape == (1, len(input_ids[0]), config.vocab_size), "Output shape mismatch for model-tokenizer integration"

def test_math_training_compatibility():
    config = VishwamaiConfig()
    model = VishwamaiModel(config)
    tokenizer_config = ConceptualTokenizerConfig(
        vocab_size=config.vocab_size,
        max_length=config.max_position_embeddings
    )
    tokenizer = ConceptualTokenizer(tokenizer_config)
    
    input_text = "Solve for x: 2x + 3 = 11"
    input_ids = tokenizer.encode(input_text)
    input_ids = torch.tensor([input_ids])
    
    logits = model(input_ids)
    assert logits.shape == (1, len(input_ids[0]), config.vocab_size), "Output shape mismatch for math training compatibility"
