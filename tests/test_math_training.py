import pytest
import torch
from torch.utils.data import DataLoader
from datasets import Dataset
from vishwamai.training import VishwamaiTrainer
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig

@pytest.fixture
def sample_gsm8k_data():
    return {
        'question': [
            "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and sells the rest at the farmers market daily for $2 per egg. How much money does she make every day at the farmers market?",
            "A robe takes 2 blue pieces of cloth and 5 white pieces of cloth. If I have 18 blue pieces and 45 white pieces, how many complete robes can I make?"
        ],
        'answer': [
            "Let's solve this step by step:\n1) Janet gets 16 eggs per day\n2) She eats 3 eggs\n3) Remaining eggs = 16 - 3 = 13\n4) Price per egg = $2\n5) Money made = 13 × $2 = $26\nAnswer: $26",
            "Let's solve this step by step:\n1) Each robe needs:\n   * 2 blue pieces\n   * 5 white pieces\n2) We have:\n   * 18 blue pieces\n   * 45 white pieces\n3) With blue pieces we can make: 18 ÷ 2 = 9 robes\n4) With white pieces we can make: 45 ÷ 5 = 9 robes\nAnswer: 9 robes"
        ]
    }

@pytest.fixture
def sample_gsm8k_dataset(sample_gsm8k_data):
    return Dataset.from_dict(sample_gsm8k_data)

@pytest.fixture
def tokenizer():
    config = ConceptualTokenizerConfig(vocab_size=32000, max_length=512)
    return ConceptualTokenizer(config)

def test_gsm8k_collate_fn(sample_gsm8k_data, tokenizer):
    from train_math import math_collate_fn
    
    batch = [
        {'question': q, 'answer': a} 
        for q, a in zip(sample_gsm8k_data['question'], sample_gsm8k_data['answer'])
    ]
    
    # Pass tokenizer to make sure we use the same one throughout the test
    collated = math_collate_fn(batch, tokenizer=tokenizer, dataset_type="gsm8k")
    
    assert 'input_ids' in collated
    assert 'concept_ids' in collated
    assert 'labels' in collated
    assert isinstance(collated['input_ids'], torch.Tensor)
    assert isinstance(collated['concept_ids'], torch.Tensor)
    assert isinstance(collated['labels'], torch.Tensor)

def test_gsm8k_dataloader(sample_gsm8k_dataset, tokenizer):
    from train_math import math_collate_fn
    from functools import partial
    
    # Use partial to bind tokenizer and dataset_type to collate_fn
    collate_fn = partial(math_collate_fn, tokenizer=tokenizer, dataset_type="gsm8k")
    
    dataloader = DataLoader(
        sample_gsm8k_dataset,
        batch_size=2,
        collate_fn=collate_fn
    )
    
    batch = next(iter(dataloader))
    assert len(batch['input_ids']) == 2
    assert len(batch['concept_ids']) == 2
    assert len(batch['labels']) == 2

def test_gsm8k_training_step(sample_gsm8k_dataset, tokenizer):
    from train_math import math_collate_fn
    from functools import partial
    from vishwamai.model import VishwamaiConfig, VishwamaiModel
    
    # Create a tiny model for testing
    config = VishwamaiConfig(
        vocab_size=32000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4
    )
    # Create model and ensure it's on CPU for testing
    model = VishwamaiModel(config).to("cpu")
    
    # Use partial to bind tokenizer and dataset_type to collate_fn
    collate_fn = partial(math_collate_fn, tokenizer=tokenizer, dataset_type="gsm8k")
    
    dataloader = DataLoader(
        sample_gsm8k_dataset,
        batch_size=2,
        collate_fn=collate_fn
    )
    
    trainer = VishwamaiTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataloader,
        eval_dataset=dataloader,
        device="cpu",  # Explicitly use CPU for testing
        optimizer_class=torch.optim.AdamW,
        use_wandb=False
    )
    
    # Test a single training step
    batch = next(iter(dataloader))
    loss = trainer.compute_loss(batch)  # compute_loss is a method of trainer that already has model
    
    assert isinstance(loss, torch.Tensor)
    assert not torch.isnan(loss)
    assert not torch.isinf(loss)

@pytest.mark.parametrize("max_length", [512])  # Use consistent max length that matches tokenizer config
def test_gsm8k_model_generation(sample_gsm8k_data, tokenizer, max_length):
    from vishwamai.model import VishwamaiConfig, VishwamaiModel
    from vishwamai.generate import VishwamaiGenerator, GenerationConfig
    
    # Create a tiny model for testing
    config = VishwamaiConfig(
        vocab_size=32000,
        hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        max_seq_len=512  # Match tokenizer max_length
    )
    model = VishwamaiModel(config)
    
    generator = VishwamaiGenerator(
        model=model,
        tokenizer=tokenizer,
        config=GenerationConfig(
            max_length=max_length,
            temperature=0.7,
            top_p=0.9
        )
    )
    
    # Test generation on a sample problem
    output = generator.generate(sample_gsm8k_data['question'][0])
    
    assert isinstance(output, list)
    assert len(output) > 0
    assert isinstance(output[0], str)
