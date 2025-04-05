"""Test suite for VishwamAI configuration components."""

import pytest
from vishwamai.configs.tpu_v3_config import TPUV3Config
from vishwamai.configs.budget_model_config import BudgetModelConfig

@pytest.fixture
def tpu_config():
    """Provide TPU configuration for testing."""
    return TPUV3Config()

@pytest.fixture
def budget_config():
    """Provide budget model configuration for testing."""
    return BudgetModelConfig()

def test_tpu_v3_config(tpu_config):
    """Test TPU V3 configuration initialization and values."""
    # Test model configuration
    assert tpu_config.model_config["vocab_size"] == 32000
    assert tpu_config.model_config["hidden_size"] == 768
    assert tpu_config.model_config["num_attention_heads"] == 12
    assert tpu_config.model_config["num_hidden_layers"] == 12
    
    # Test training configuration
    assert "batch_size" in tpu_config.training_config
    assert "grad_accum_steps" in tpu_config.training_config
    assert "learning_rate" in tpu_config.training_config
    
    # Test TPU-specific configuration
    assert "mesh_shape" in tpu_config.tpu_config
    assert tpu_config.tpu_config["precision"] == "bfloat16"
    
    # Test memory optimizations
    assert tpu_config.memory_config["use_gradient_checkpointing"]
    assert tpu_config.memory_config["use_memory_efficient_attention"]

def test_budget_model_config(budget_config):
    """Test budget model configuration for limited resources."""
    # Test reduced model size
    assert budget_config.model_config["hidden_size"] < 1024, \
        "Budget model hidden size should be smaller than 1024"
    assert budget_config.model_config["num_hidden_layers"] < 12, \
        "Budget model should have fewer layers"
    
    # Test memory optimizations are enabled
    assert budget_config.memory_config["use_gradient_checkpointing"], \
        "Gradient checkpointing should be enabled for budget model"
    assert budget_config.memory_config["use_fp8_kv_cache"], \
        "FP8 KV cache should be enabled for budget model"
    
    # Test effective batch size calculation
    batch_size = budget_config.get_effective_batch_size()
    assert batch_size <= 256, \
        f"Effective batch size {batch_size} too large for budget model"

def test_config_compatibility(tpu_config, budget_config):
    """Test configuration compatibility across components."""
    # Verify TPU mesh configuration
    assert len(tpu_config.tpu_config["mesh_shape"]) == 1, \
        "Should use 1D mesh for data parallel training"
    
    # Check training configuration alignment
    assert tpu_config.training_config["gradient_accumulation_steps"] >= \
           budget_config.training_config["gradient_accumulation_steps"], \
           "TPU config should have >= gradient accumulation steps than budget config"
    
    # Validate memory configurations
    assert tpu_config.memory_config["use_gradient_checkpointing"] == \
           budget_config.memory_config["use_gradient_checkpointing"], \
           "Memory optimization settings should match"
