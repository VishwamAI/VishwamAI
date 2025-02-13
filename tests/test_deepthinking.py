import pytest
import torch
from vishwamai.deepthinking import CoTGenerationWrapper, GRPOTrainer, ReasoningDataset
from vishwamai.model import VishwamaiModel
from vishwamai.conceptual_tokenizer import ConceptualTokenizer

from tests.config_fixtures import VishwamaiConfig, TokenizerConfig

@pytest.fixture
def model_config():
    return VishwamaiConfig.get_test_config()

@pytest.fixture
def tokenizer(model_config):
    # Mock tokenizer for testing
    class MockTokenizer:
        def __init__(self, device):
            self.config = TokenizerConfig()
            self.device = device
            self.concept_embeddings = {
                "[CONCEPT_THINK_START]": 1,
                "[CONCEPT_THINK_END]": 2,
                "[CONCEPT_ANSWER_START]": 3,
                "[CONCEPT_ANSWER_END]": 4
            }
            self.unk_id = 0
            
        def encode(self, text, add_special_tokens=True):
            return [0] * 10  # Mock encoding
            
        def decode(self, ids):
            return "mock decoded text"
            
        def batch_encode_with_concepts(self, texts):
            batch_size = len(texts)
            return {
                'input_ids': torch.ones(batch_size, 10, device=self.device).long(),
                'concept_ids': torch.zeros(batch_size, 10, device=self.device).long()
            }
    
    return MockTokenizer(model_config.torch_device)

@pytest.fixture
def model(model_config):
    return VishwamaiModel(model_config)

def test_cot_generation_wrapper(model_config, tokenizer):
    model = VishwamaiModel(model_config)
    wrapper = CoTGenerationWrapper(model, tokenizer)
    
    # Move to configured device
    wrapper.to(model_config.torch_device)
    
    prompt = "What is 2+2?"
    output = wrapper.generate(prompt, max_new_tokens=20)
    
    assert isinstance(output, list)
    assert len(output) > 0
    assert all(isinstance(seq, dict) for seq in output)
    assert all('thought' in seq and 'answer' in seq for seq in output)
    assert all(isinstance(seq['thought'], str) for seq in output)
    assert all(isinstance(seq['answer'], str) for seq in output)

def test_grpo_trainer_initialization(model_config, tokenizer):
    model = VishwamaiModel(model_config)
    reward_fns = {
        'format': lambda x: 1.0,
        'accuracy': lambda x: 1.0
    }
    
    def mock_reward(x): return 1.0
    reward_fns = {
        'format': mock_reward,
        'accuracy': mock_reward,
        'reflection': mock_reward
    }
    
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fns=reward_fns,
        group_size=model_config.group_size,
        beta=model_config.beta
    )
    
    # Move to configured device
    trainer.model.to(model_config.torch_device)
    trainer.ref_model.to(model_config.torch_device)
    
    assert isinstance(trainer.model, VishwamaiModel)
    assert isinstance(trainer.ref_model, VishwamaiModel)
    assert trainer.gamma == 0.99
    assert trainer.eps_clip == 0.2

def test_reasoning_dataset(tokenizer):
    problems = [
        {
            'question': 'What is 2+2?',
            'thought': 'I need to add two numbers',
            'answer': '4',
            'concepts': [1, 2, 3]
        }
    ]
    
    dataset = ReasoningDataset(problems, tokenizer)
    assert len(dataset) == 1
    
    item = dataset[0]
    assert 'input_ids' in item
    assert 'concept_ids' in item

def test_reward_computation(model_config, tokenizer):
    model = VishwamaiModel(model_config)
    
    def mock_reward(x): return 1.0
    reward_fns = {
        'format': mock_reward,
        'accuracy': mock_reward
    }
    
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fns=reward_fns,
        group_size=model_config.group_size,
        beta=model_config.beta
    )
    
    # Move to configured device
    trainer.model.to(model_config.torch_device)
    trainer.ref_model.to(model_config.torch_device)
    
    responses = ["Test response 1", "Test response 2"]
    rewards = trainer.compute_rewards(responses)
    
    assert isinstance(rewards, dict)
    assert 'format' in rewards
    assert 'accuracy' in rewards
    assert all(isinstance(r, torch.Tensor) for r in rewards.values())

def test_self_reflection(model_config, tokenizer):
    model = VishwamaiModel(model_config)
    wrapper = CoTGenerationWrapper(model, tokenizer)
    wrapper.to(model_config.torch_device)
    
    # Test self-reflection capability
    prompt = "Solve this step by step: If a train travels 120 km in 2 hours, what is its speed?"
    output = wrapper.generate(prompt, max_new_tokens=100)
    
    # Check for structured thinking pattern
    assert len(output) > 0
    response = output[0]
    assert 'thought' in response
    assert 'answer' in response
    
    # Verify thought process includes key components
    thought = response['thought']
    answer = response['answer']
    assert isinstance(thought, str)
    assert isinstance(answer, str)

@pytest.mark.parametrize("num_reflect_steps", [1, 2, 3])
def test_reflection_steps(model_config, tokenizer, num_reflect_steps):
    model = VishwamaiModel(model_config)
    wrapper = CoTGenerationWrapper(model, tokenizer, num_self_reflect_steps=num_reflect_steps)
    wrapper.to(model_config.torch_device)
    
    output = wrapper.generate("Test reflection steps", max_new_tokens=100)
    
    assert len(output) > 0
    reflections = output[0]['reflections']
    assert len(reflections) <= num_reflect_steps
    for reflection in reflections:
        assert reflection.startswith('[Reflection')

@pytest.mark.parametrize("invalid_token_id", [None, -1, 99999])
def test_special_token_fallbacks(model_config, tokenizer, invalid_token_id):
    model = VishwamaiModel(model_config)
    orig_embeddings = tokenizer.concept_embeddings.copy()
    
    # Set invalid tokens
    tokenizer.concept_embeddings = {
        "[CONCEPT_THINK_START]": invalid_token_id,
        "[CONCEPT_THINK_END]": invalid_token_id
    }
    
    wrapper = CoTGenerationWrapper(model, tokenizer)
    wrapper.to(model_config.torch_device)
    
    # Check fallback behavior
    assert hasattr(wrapper, 'think_start_id')
    assert hasattr(wrapper, 'think_end_id')
    assert wrapper.think_start_id == tokenizer.unk_id
    assert wrapper.think_end_id == tokenizer.unk_id
    
    # Restore original embeddings
    tokenizer.concept_embeddings = orig_embeddings

def test_temperature_sampling(model_config, tokenizer):
    model = VishwamaiModel(model_config)
    wrapper = CoTGenerationWrapper(model, tokenizer)
    wrapper.to(model_config.torch_device)
    
    # Generate with different temperatures
    prompt = "Test temperature"
    output_high_temp = wrapper.generate(prompt, temperature=1.0)
    output_low_temp = wrapper.generate(prompt, temperature=0.1)
    
    assert len(output_high_temp) > 0
    assert len(output_low_temp) > 0
    # Lower temperature should lead to more focused outputs
    assert len(output_low_temp[0]['thought']) <= len(output_high_temp[0]['thought'])

def test_top_p_filtering(model_config, tokenizer):
    model = VishwamaiModel(model_config)
    wrapper = CoTGenerationWrapper(model, tokenizer)
    wrapper.to(model_config.torch_device)
    
    # Generate with different top_p values 
    prompt = "Test top_p"
    output_high_p = wrapper.generate(prompt, top_p=0.9)
    output_low_p = wrapper.generate(prompt, top_p=0.1)
    
    assert len(output_high_p) > 0 
    assert len(output_low_p) > 0
    # Lower top_p should lead to more conservative outputs
    assert len(output_low_p[0]['thought']) <= len(output_high_p[0]['thought'])

def test_batch_generation(model_config, tokenizer):
    model = VishwamaiModel(model_config)
    wrapper = CoTGenerationWrapper(model, tokenizer)
    wrapper.to(model_config.torch_device)
    prompts = ["Question 1", "Question 2", "Question 3"]
    num_samples = 2
    
    all_outputs = []
    for prompt in prompts:
        outputs = wrapper.generate(prompt, num_samples=num_samples)
        all_outputs.extend(outputs)
    
    assert len(all_outputs) == len(prompts) * num_samples
    assert all(isinstance(out, dict) for out in all_outputs)

def test_reward_scaling(model_config, tokenizer):
    model = VishwamaiModel(model_config)
    
    # Create mock rewards that return tensors on the correct device
    def mock_reward(x): 
        return torch.tensor(0.5, device=model_config.torch_device)
        
    reward_fns = {
        'accuracy': mock_reward,
        'coherence': mock_reward,
        'step_following': mock_reward
    }
    
    trainer = GRPOTrainer(
        model=model,
        tokenizer=tokenizer,
        reward_fns=reward_fns,
        group_size=model_config.group_size,
        beta=model_config.beta
    )
    
    trainer.model.to(model_config.torch_device)
    trainer.ref_model.to(model_config.torch_device)
    
    responses = ["Response 1", "Response 2", "Response 3"]
    rewards = trainer.compute_rewards(responses)
    
    # Move tensors to CPU for comparison
    for reward_type, values in rewards.items():
        cpu_values = values.cpu()
        assert torch.abs(cpu_values.mean()) < 1e-6  # Should be close to 0
        assert torch.abs(cpu_values.std() - 1.0) < 1e-6  # Should be close to 1

def test_model_saves_and_loads(tmp_path, model_config, tokenizer):
    model = VishwamaiModel(model_config)
    wrapper = CoTGenerationWrapper(model, tokenizer)
    wrapper.to(model_config.torch_device)
    save_path = tmp_path / "test_model.pt"
    
    # Save model
    torch.save(wrapper.state_dict(), save_path)
    
    # Load model
    new_model = VishwamaiModel(model_config)
    new_wrapper = CoTGenerationWrapper(new_model, tokenizer)
    new_wrapper.to(model_config.torch_device)
    new_wrapper.load_state_dict(torch.load(save_path))
    
    # Test both produce similar outputs
    prompt = "Test prompt"
    with torch.no_grad():
        output1 = wrapper.generate(prompt)
        output2 = new_wrapper.generate(prompt)
    
    assert len(output1) == len(output2)
    for o1, o2 in zip(output1, output2):
        assert o1['thought'] == o2['thought']
        assert o1['answer'] == o2['answer']
