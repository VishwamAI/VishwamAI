import pytest
import torch
from vishwamai.model import VishwamaiModel, VishwamaiConfig
from vishwamai.generate import VishwamaiGenerator, GenerationConfig
from vishwamai.conceptual_tokenizer import ConceptualTokenizer

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
    tokenizer = ConceptualTokenizer(vocab_size=100, max_length=32)
    return tokenizer

@pytest.fixture
def generator(model, tokenizer):
    gen_config = GenerationConfig(
        max_length=10,
        temperature=0.7,
        top_p=0.9,
        top_k=50,
        num_return_sequences=1
    )
    return VishwamaiGenerator(model, tokenizer, gen_config)

def test_generate_basic(generator):
    prompt = "Test input"
    output = generator.generate(prompt)
    assert isinstance(output, list), "Output is not a list"
    assert len(output) == 1, "Number of generated sequences mismatch"
    assert isinstance(output[0], str), "Generated output is not a string"
    assert len(output[0]) > 0, "Generated output is empty"

def test_generate_with_attention_mask(generator):
    prompt = "Test with attention mask"
    input_ids = generator.tokenizer.encode(prompt, return_tensors="pt")
    attention_mask = torch.ones_like(input_ids)
    
    with torch.no_grad():
        output_ids = generator._generate_tokens(input_ids)
    
    assert output_ids.shape[1] <= generator.config.max_length, "Generated sequence exceeds max length"

def test_generate_temperature_variation(model, tokenizer):
    gen_config_high_temp = GenerationConfig(
        max_length=10,
        temperature=2.0,
        top_p=0.9,
        top_k=50,
        num_return_sequences=1
    )
    generator_high = VishwamaiGenerator(model, tokenizer, gen_config_high_temp)
    
    gen_config_low_temp = GenerationConfig(
        max_length=10,
        temperature=0.5,
        top_p=0.9,
        top_k=50,
        num_return_sequences=1
    )
    generator_low = VishwamaiGenerator(model, tokenizer, gen_config_low_temp)
    
    prompt = "Temperature test"
    output_high = generator_high.generate(prompt)
    output_low = generator_low.generate(prompt)
    
    assert output_high != output_low, "Outputs should differ with different temperatures"

def test_generate_top_p_variation(model, tokenizer):
    gen_config_high_p = GenerationConfig(
        max_length=10,
        temperature=1.0,
        top_p=0.95,
        top_k=50,
        num_return_sequences=1
    )
    generator_high_p = VishwamaiGenerator(model, tokenizer, gen_config_high_p)
    
    gen_config_low_p = GenerationConfig(
        max_length=10,
        temperature=1.0,
        top_p=0.5,
        top_k=50,
        num_return_sequences=1
    )
    generator_low_p = VishwamaiGenerator(model, tokenizer, gen_config_low_p)
    
    prompt = "Top-p test"
    output_high_p = generator_high_p.generate(prompt)
    output_low_p = generator_low_p.generate(prompt)
    
    assert output_high_p != output_low_p, "Outputs should differ with different top-p values"

def test_generate_with_eos(generator):
    prompt = "End of sequence test"
    # Assuming tokenizer has an EOS token
    output = generator.generate(prompt)
    assert output[0].endswith("</s>"), "Output does not end with EOS token"

def test_generate_max_length(generator):
    prompt = "Max length test"
    gen_config = GenerationConfig(
        max_length=5,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        num_return_sequences=1
    )
    generator_limited = VishwamaiGenerator(generator.model, generator.tokenizer, gen_config)
    output = generator_limited.generate(prompt)
    encoded_length = len(generator_limited.tokenizer.encode(prompt, return_tensors="pt")[0]) + 5
    assert len(generator_limited.tokenizer.encode(output[0])) <= encoded_length, "Generated sequence exceeds max length"
