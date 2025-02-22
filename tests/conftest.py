"""
Test fixtures and configurations
"""
import pytest
import torch
from typing import Dict, Any, List

from vishwamai.config import (
    ModelConfig,
    PrecisionConfig,
    PrecisionMode,
    TreePlannerConfig,
    InformationRetrievalConfig
)

def create_test_config(
    precision_mode: PrecisionMode = PrecisionMode.FP16,
    mixed_precision: bool = True,
    gradient_precision: str = "fp32"
) -> ModelConfig:
    """Create a test model configuration"""
    return ModelConfig(
        hidden_size=128,
        num_layers=2,
        num_heads=4,
        intermediate_size=512,
        vocab_size=1000,
        precision=PrecisionConfig(
            mode=precision_mode,
            mixed_precision=mixed_precision,
            gradient_precision=gradient_precision
        ),
        tree_planner=TreePlannerConfig(
            enabled=True,
            num_tree_layers=2,
            tree_hidden_size=64
        ),
        information_retrieval=InformationRetrievalConfig(
            enabled=True,
            max_queries_per_response=2
        )
    )

@pytest.fixture
def device():
    """Get the appropriate device"""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture
def base_config() -> ModelConfig:
    """Get base model configuration"""
    return create_test_config()

@pytest.fixture
def mixed_precision_config() -> ModelConfig:
    """Get mixed precision configuration"""
    return create_test_config(
        precision_mode=PrecisionMode.FP16,
        mixed_precision=True
    )

@pytest.fixture
def fp32_config() -> ModelConfig:
    """Get FP32 configuration"""
    return create_test_config(
        precision_mode=PrecisionMode.FP32,
        mixed_precision=False
    )

@pytest.fixture
def fp64_config() -> ModelConfig:
    """Get FP64 configuration"""
    return create_test_config(
        precision_mode=PrecisionMode.FP64,
        mixed_precision=False,
        gradient_precision="fp64"
    )

@pytest.fixture
def precision_test_cases() -> List[Dict[str, Any]]:
    """Get precision test cases"""
    return [
        {
            "mode": PrecisionMode.FP16,
            "mixed_precision": True,
            "gradient_precision": "fp32",
            "expected_dtype": torch.float16
        },
        {
            "mode": PrecisionMode.FP32,
            "mixed_precision": False,
            "gradient_precision": "fp32",
            "expected_dtype": torch.float32
        },
        {
            "mode": PrecisionMode.FP64,
            "mixed_precision": False,
            "gradient_precision": "fp64",
            "expected_dtype": torch.float64
        }
    ]

@pytest.fixture
def small_batch(device) -> Dict[str, torch.Tensor]:
    """Create a small batch for testing"""
    return {
        "input_ids": torch.randint(0, 1000, (4, 16)).to(device),
        "attention_mask": torch.ones(4, 16).to(device)
    }

@pytest.fixture
def memory_tracker():
    """Track GPU memory usage"""
    class MemoryTracker:
        def __init__(self):
            self.reset()
            
        def reset(self):
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
                self.start_memory = torch.cuda.memory_allocated()
            else:
                self.start_memory = 0
                
        def get_max_memory(self) -> int:
            if torch.cuda.is_available():
                return torch.cuda.max_memory_allocated() - self.start_memory
            return 0
            
        def get_current_memory(self) -> int:
            if torch.cuda.is_available():
                return torch.cuda.memory_allocated() - self.start_memory
            return 0
            
    return MemoryTracker()

@pytest.fixture
def model_factory(device):
    """Create models with different configurations"""
    from vishwamai.model import VishwamaiModel
    
    def create_model(config: ModelConfig) -> VishwamaiModel:
        model = VishwamaiModel(config)
        return model.to(device)
        
    return create_model

def pytest_addoption(parser):
    """Add custom command line options"""
    parser.addoption(
        "--precision-mode",
        choices=["fp16", "fp32", "fp64", "bf16"],
        default="fp16",
        help="Precision mode for testing"
    )

@pytest.fixture
def precision_mode(request) -> PrecisionMode:
    """Get precision mode from command line"""
    return PrecisionMode(request.config.getoption("--precision-mode"))
