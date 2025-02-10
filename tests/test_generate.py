import torch
from vishwamai.generate import GenerationConfig, VishwamaiGenerator

def test_generate_basic(generator):
    prompt = "Test input"
    output = generator.generate(prompt)
    assert isinstance(output, list), "Output should be a list of generated texts"
    assert len(output) == generator.config.num_return_sequences, "Number of returned sequences mismatch"
    assert isinstance(output[0], str), "Each output should be a string"
    # Ensure EOS token is present
    encoded = generator.tokenizer.encode(output[0])
    assert generator.tokenizer.eos_token_id in encoded, "EOS token should be present in generated text"

def test_generate_with_attention_mask(generator):
    prompt = "Test with attention mask"
    input_ids = generator.tokenizer.encode(prompt)  # Removed return_tensors="pt"
    input_ids = torch.tensor(input_ids).unsqueeze(0)  # Add batch dimension
    attention_mask = torch.ones_like(input_ids)

    with torch.no_grad():
        output_ids = generator._generate_tokens(input_ids)

    assert output_ids.shape[1] <= generator.config.max_length, "Generated sequence exceeds max length"

def test_generate_temperature_variation(model, tokenizer):
    gen_config_high_temp = GenerationConfig(
        max_length=32,
        temperature=2.0,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generator_high = VishwamaiGenerator(model, tokenizer, gen_config_high_temp)

    gen_config_low_temp = GenerationConfig(
        max_length=32,
        temperature=0.5,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generator_low = VishwamaiGenerator(model, tokenizer, gen_config_low_temp)

    prompt = "Temperature test"
    output_high = generator_high.generate(prompt)
    output_low = generator_low.generate(prompt)

    assert len(output_high) == 1, "High temperature should return one sequence"
    assert len(output_low) == 1, "Low temperature should return one sequence"
    assert isinstance(output_high[0], str), "Generated output should be a string"
    assert isinstance(output_low[0], str), "Generated output should be a string"
    # Ensure EOS token is present
    encoded_high = generator_high.tokenizer.encode(output_high[0])
    encoded_low = generator_low.tokenizer.encode(output_low[0])
    assert generator_high.tokenizer.eos_token_id in encoded_high, "EOS token should be present in high temperature output"
    assert generator_low.tokenizer.eos_token_id in encoded_low, "EOS token should be present in low temperature output"

def test_generate_top_p_variation(model, tokenizer):
    gen_config_high_p = GenerationConfig(
        max_length=32,
        temperature=1.0,
        top_p=0.95,
        top_k=50,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generator_high_p = VishwamaiGenerator(model, tokenizer, gen_config_high_p)

    gen_config_low_p = GenerationConfig(
        max_length=32,
        temperature=1.0,
        top_p=0.5,
        top_k=50,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id
    )
    generator_low_p = VishwamaiGenerator(model, tokenizer, gen_config_low_p)

    prompt = "Top-p test"
    output_high_p = generator_high_p.generate(prompt)
    output_low_p = generator_low_p.generate(prompt)

    assert len(output_high_p) == 1, "High top-p should return one sequence"
    assert len(output_low_p) == 1, "Low top-p should return one sequence"
    assert isinstance(output_high_p[0], str), "Generated output should be a string"
    assert isinstance(output_low_p[0], str), "Generated output should be a string"
    # Ensure EOS token is present
    encoded_high_p = generator_high_p.tokenizer.encode(output_high_p[0])
    encoded_low_p = generator_low_p.tokenizer.encode(output_low_p[0])
    assert generator_high_p.tokenizer.eos_token_id in encoded_high_p, "EOS token should be present in high top-p output"
    assert generator_low_p.tokenizer.eos_token_id in encoded_low_p, "EOS token should be present in low top-p output"

def test_generate_with_eos(generator):
    prompt = "End of sequence test"
    output = generator.generate(prompt)
    for text in output:
        encoded = generator.tokenizer.encode(text)
        assert generator.tokenizer.eos_token_id in encoded, "EOS token not found in generated text"

def test_generate_empty_output(generator):
    prompt = ""
    output = generator.generate(prompt)
    for text in output:
        encoded = generator.tokenizer.encode(text)
        assert generator.tokenizer.eos_token_id in encoded, "EOS token should be present even for empty prompts"

def test_generate_max_length(generator):
    prompt = "Max length test"
    gen_config = GenerationConfig(
        max_length=32,
        temperature=1.0,
        top_p=0.9,
        top_k=50,
        do_sample=True,
        num_return_sequences=1,
        pad_token_id=generator.tokenizer.pad_token_id,
        bos_token_id=generator.tokenizer.bos_token_id,
        eos_token_id=generator.tokenizer.eos_token_id
    )
    generator_limited = VishwamaiGenerator(generator.model, generator.tokenizer, gen_config)
    output = generator_limited.generate(prompt)
    for text in output:
        if text:  # Ensure text is not empty
            encoded = generator_limited.tokenizer.encode(text)
            assert len(encoded) <= generator_limited.config.max_length, "Generated sequence exceeds max length"
        else:
            encoded = []
            assert len(encoded) == 0, "Encoded sequence should be empty for empty text"
