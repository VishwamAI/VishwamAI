import pytest
import torch
import torch.nn as nn
from vishwamai.deep_reasoning import EnhancedReasoning, ReasoningStep, ReasoningOutput

from tests.config_fixtures import VishwamaiConfig, TokenizerConfig

@pytest.fixture
def config():
    return VishwamaiConfig.get_test_config()

@pytest.fixture
def reasoning_module(config):
    return EnhancedReasoning(config)

@pytest.fixture
def sample_input():
    batch_size = 2
    seq_len = 10
    hidden_size = 128
    return {
        'hidden_states': torch.randn(batch_size, seq_len, hidden_size),
        'attention_mask': torch.ones(batch_size, seq_len),
        'knowledge_context': torch.randn(batch_size, seq_len, hidden_size)
    }

def test_reasoning_step_initialization():
    step = ReasoningStep(
        premise=torch.randn(1, 128),
        inference=torch.randn(1, 128),
        confidence=0.8,
        step_num=1,
        reflection="Test reflection"
    )
    
    assert isinstance(step.premise, torch.Tensor)
    assert isinstance(step.inference, torch.Tensor)
    assert 0 <= step.confidence <= 1
    assert step.step_num == 1
    assert step.reflection == "Test reflection"

def test_enhanced_reasoning_initialization(reasoning_module):
    assert isinstance(reasoning_module.structured_reasoning, nn.Module)
    assert isinstance(reasoning_module.verification, nn.Module)
    assert isinstance(reasoning_module.metacognition, nn.Module)
    assert isinstance(reasoning_module.step_encoder, nn.Linear)
    assert isinstance(reasoning_module.reflection_generator, nn.Sequential)

def test_generate_reflection(reasoning_module):
    current_state = torch.randn(1, 128)
    reasoning_history = torch.randn(1, 128)
    
    reflection, confidence = reasoning_module.generate_reflection(
        current_state,
        reasoning_history
    )
    
    assert isinstance(reflection, torch.Tensor)
    assert reflection.shape == (1, 128)
    assert isinstance(confidence, float)
    assert 0 <= confidence <= 1

def test_reasoning_forward_pass(reasoning_module, sample_input):
    output = reasoning_module(**sample_input)
    
    assert isinstance(output, ReasoningOutput)
    assert len(output.steps) > 0
    assert isinstance(output.verification, dict)
    assert isinstance(output.meta_analysis, dict)
    assert isinstance(output.reflections, list)
    assert isinstance(output.final_answer, list)
    assert isinstance(output.confidence, float)
    assert isinstance(output.trace, dict)

def test_reflection_generation_threshold(reasoning_module, sample_input):
    # Test with high reflection threshold
    output_high = reasoning_module(
        **sample_input,
        reflection_threshold=0.9
    )
    
    # Test with low reflection threshold
    output_low = reasoning_module(
        **sample_input,
        reflection_threshold=0.1
    )
    
    # Should generate more reflections with lower threshold
    assert len(output_low.reflections) >= len(output_high.reflections)

def test_max_steps_limit(reasoning_module, sample_input):
    max_steps = 3
    output = reasoning_module(
        **sample_input,
        max_steps=max_steps
    )
    
    assert len(output.steps) <= max_steps

def test_analyze_reasoning_trace(reasoning_module, sample_input):
    output = reasoning_module(**sample_input)
    analysis = reasoning_module.analyze_reasoning_trace(output)
    
    expected_metrics = {
        "avg_step_confidence",
        "verification_score",
        "meta_confidence",
        "num_reflections",
        "final_confidence"
    }
    
    assert set(analysis.keys()) == expected_metrics
    assert all(isinstance(v, float) for v in analysis.values())

def test_format_reasoning_output(reasoning_module, sample_input):
    output = reasoning_module(**sample_input)
    formatted = reasoning_module.format_reasoning_output(output)
    
    assert isinstance(formatted, str)
    assert "<think>" in formatted
    assert "</think>" in formatted
    assert "Step" in formatted
    assert "Final Answer:" in formatted

def test_confidence_driven_early_stopping(reasoning_module, sample_input):
    # Test that reasoning stops early when high confidence is reached
    output = reasoning_module(
        **sample_input,
        max_steps=10,  # High max steps
        reflection_threshold=0.5
    )
    
    # Check if process stopped before max_steps if high confidence was reached
    final_confidence = output.steps[-1].confidence
    if final_confidence > 0.95:
        assert len(output.steps) < 10

def test_reasoning_with_no_knowledge_context(reasoning_module, sample_input):
    # Test reasoning without external knowledge context
    sample_input_no_context = {
        'hidden_states': sample_input['hidden_states'],
        'attention_mask': sample_input['attention_mask']
    }
    
    output = reasoning_module(**sample_input_no_context)
    assert isinstance(output, ReasoningOutput)
    assert len(output.steps) > 0

@pytest.mark.parametrize("batch_size", [1, 2, 4])
def test_batch_processing(reasoning_module, config, batch_size):
    seq_len = 10
    hidden_size = config.hidden_size
    
    input_batch = {
        'hidden_states': torch.randn(batch_size, seq_len, hidden_size),
        'attention_mask': torch.ones(batch_size, seq_len)
    }
    
    output = reasoning_module(**input_batch)
    assert isinstance(output.final_answer, list)
    assert len(output.final_answer) == batch_size

def test_reasoning_trace_consistency(reasoning_module, sample_input):
    output = reasoning_module(**sample_input)
    
    # Check trace lengths match number of steps
    assert len(output.trace["step_confidence"]) == len(output.steps)
    assert len(output.trace["verification_scores"]) > 0
    assert len(output.trace["meta_confidence"]) > 0
    
    # Check confidence values are valid
    assert all(0 <= c <= 1 for c in output.trace["step_confidence"])
    assert all(0 <= c <= 1 for c in output.trace["meta_confidence"])

def test_reasoning_with_attention_mask(reasoning_module):
    batch_size = 2
    seq_len = 10
    hidden_size = 128
    
    # Create a sample input with masked tokens
    hidden_states = torch.randn(batch_size, seq_len, hidden_size)
    attention_mask = torch.zeros(batch_size, seq_len)
    attention_mask[:, :5] = 1  # Only first 5 tokens are attended to
    
    output = reasoning_module(
        hidden_states=hidden_states,
        attention_mask=attention_mask
    )
    
    assert isinstance(output, ReasoningOutput)
    assert len(output.steps) > 0

def test_model_save_load(reasoning_module, tmp_path, sample_input):
    # Get output from original model
    original_output = reasoning_module(**sample_input)
    
    # Save model
    save_path = tmp_path / "reasoning_model.pt"
    torch.save(reasoning_module.state_dict(), save_path)
    
    # Load model
    new_reasoning = EnhancedReasoning(reasoning_module.config)
    new_reasoning.load_state_dict(torch.load(save_path))
    
    # Get output from loaded model
    loaded_output = new_reasoning(**sample_input)
    
    # Compare outputs
    assert len(original_output.steps) == len(loaded_output.steps)
    assert original_output.confidence == pytest.approx(loaded_output.confidence)
