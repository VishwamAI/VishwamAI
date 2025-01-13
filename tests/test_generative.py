# #import pytest
# #import torch
# #import gc
# #from vishwamai.architecture import VishwamaiConfig, init_model
# 
# # Updated test configuration with compatible dimensions
# #TEST_CONFIG = {
# #    'dim': 256,
# #    'n_layers': 6,
# #    'n_heads': 8,
# #    'vocab_size': 1000,
# #    'max_seq_len': 512,
# #    'n_kv_heads': 8  # Added to ensure compatibility
# #}
# 
# #@pytest.fixture
# #def setup_teardown():
# #    # Setup
# #    torch.manual_seed(42)
# #    torch.use_deterministic_algorithms(True)
# #    torch.backends.cudnn.benchmark = False
# #    torch.backends.cudnn.deterministic = True
# #    if torch.cuda.is_available():
# #        torch.cuda.empty_cache()
# #    
# #    yield
# #    
# #    # Teardown
# #    if torch.cuda.is_available():
# #        torch.cuda.empty_cache()
# #    torch.use_deterministic_algorithms(False)
# #    torch.backends.cudnn.benchmark = True
# #    torch.backends.cudnn.deterministic = False
# #    gc.collect()
# #    
# #def test_model_alignment(setup_teardown):
# #    config = VishwamaiConfig(**TEST_CONFIG)
# #    
# #    try:
# #        model = init_model(config)
# #        model.eval()
# #        
# #        # Use smaller sequence length
# #        sequence_length = 32
# #        prompt = torch.randint(0, config.vocab_size, (1, sequence_length))
# #        
# #        # Generate with controlled parameters
# #        with torch.no_grad():
# #            output = model.generate(
# #                prompt,
# #                max_length=min(50, config.max_seq_len),  # Ensure max_length doesn't exceed max_seq_len
# #                do_sample=True,
# #                temperature=0.7,  # Add temperature for more stable sampling
# #                top_k=50,        # Add top_k to prevent dimension issues
# #                start_pos=0      # Explicitly set start_pos
# #            )
# #            
# #            # Verify output dimensions and content
# #            assert output.size(0) == 1, "Batch size should be 1"
# #            assert output.size(1) <= config.max_seq_len, "Generated sequence exceeds max_seq_len"
# #            assert torch.all(output < config.vocab_size), "Generated tokens exceed vocab_size"
# #            
# #    finally:
# #        del model
# #        gc.collect()
# #    
# #def test_response_consistency(setup_teardown):
# #    config = VishwamaiConfig(**TEST_CONFIG)
# #    
# #    try:
# #        model = init_model(config)
# #        model.eval()
# #        
# #        # Test with fixed sequence length
# #        sequence_length = 16
# #        responses = []
# #        
# #        with torch.no_grad():
# #            for _ in range(3):  # Generate multiple responses
# #                prompt = torch.randint(0, config.vocab_size, (1, sequence_length))
# #                response = model.generate(
# #                    prompt,
# #                    max_length=min(32, config.max_seq_len),
# #                    do_sample=True,
# #                    temperature=0.7,
# #                    top_k=50
# #                )
# #                responses.append(response)
# #                
# #                # Verify each response
# #                assert response.size(1) <= config.max_seq_len, "Response exceeds max_seq_len"
# #                assert torch.all(response < config.vocab_size), "Response contains invalid tokens"
# #        
# #        # Verify diversity
# #        assert len(set(tuple(r.flatten().tolist()) for r in responses)) > 1, "All responses are identical"
# #        
# #    finally:
# #        del model
# #        gc.collect()
# #    
# #def test_multilingual_capabilities(setup_teardown):
# #    config = VishwamaiConfig(**TEST_CONFIG)
# #    
# #    try:
# #        model = init_model(config)
# #        model.eval()
# #        
# #        test_prompts = {
# #            "english": "Hello, how are you?",
# #            "spanish": "¿Hola, cómo estás?",
# #            "chinese": "你好，你好吗？",
# #            "hindi": "नमस्ते, आप कैसे हैं?"
# #        }
# #        
# #        for language, prompt in test_prompts.items():
# #            response = model.generate(torch.randint(0, config.vocab_size, (1, len(prompt))), max_length=50, do_sample=True)
# #            assert response.size(1) <= 50, f"Generated sequence exceeds max_length for {language}"
# #            assert torch.all(response < config.vocab_size), f"Generated tokens exceed vocab_size for {language}"
# #            
# #    finally:
# #        del model
# #        gc.collect()
# #    
# #def test_model_safety(setup_teardown):
# #    config = VishwamaiConfig(**TEST_CONFIG)
# #    
# #    try:
# #        model = init_model(config)
# #        model.eval()
# #        
# #        sequence_length = 16
# #        prompt = torch.randint(0, config.vocab_size, (1, sequence_length))
# #        
# #        with torch.no_grad():
# #            response = model.generate_with_safety(
# #                prompt,
# #                max_length=min(32, config.max_seq_len),
# #                do_sample=True,
# #                temperature=0.7,
# #                top_k=50
# #            )
# #            
# #            # Verify output dimensions and content
# #            assert response.size(0) == 1, "Batch size should be 1"
# #            assert response.size(1) <= config.max_seq_len, "Generated sequence exceeds max_seq_len"
# #            assert torch.all(response < config.vocab_size), "Generated tokens exceed vocab_size"
# #        
# #    finally:
# #        del model
# #        gc.collect()
# #    
# # Helper function for semantic similarity check
# #def verify_semantic_similarity(responses):
# #    # Implement semantic similarity check logic
# #    # For now, just check if responses are different
# #    return len(set(tuple(r.flatten().tolist()) for r in responses)) > 1
