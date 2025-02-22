"""
Tests for tree planning components with different precision modes
"""
import pytest
import torch
import numpy as np

from vishwamai.config import ModelConfig, PrecisionConfig, PrecisionMode
from vishwamai.model import VishwamaiModel
from vishwamai.tree_planner import TreePlanner
from vishwamai.training.optimization import GradScaler

@pytest.mark.precision
@pytest.mark.gpu
def test_tree_planner_precision_modes(model_factory, precision_test_cases, device):
    """Test tree planner with different precision modes"""
    for test_case in precision_test_cases:
        # Configure model with specific precision
        config = ModelConfig(
            hidden_size=128,
            num_layers=2,
            precision=PrecisionConfig(
                mode=test_case["mode"],
                mixed_precision=test_case["mixed_precision"],
                gradient_precision=test_case["gradient_precision"]
            ),
            tree_planner={
                "enabled": True,
                "num_tree_layers": 2,
                "tree_hidden_size": 64
            }
        )
        
        model = model_factory(config)
        
        # Check tree planner parameters precision
        tree_planner = model.tree_planner
        for param in tree_planner.parameters():
            assert param.dtype == test_case["expected_dtype"]

@pytest.mark.precision
@pytest.mark.gpu
def test_tree_planning_with_retrieval(model_factory, small_batch, device):
    """Test tree planning with information retrieval under different precisions"""
    for mode in [PrecisionMode.FP16, PrecisionMode.FP32]:
        config = ModelConfig(
            hidden_size=128,
            num_layers=2,
            precision=PrecisionConfig(mode=mode),
            tree_planner={
                "enabled": True,
                "num_tree_layers": 2,
                "tree_hidden_size": 64
            },
            information_retrieval={
                "enabled": True,
                "search_query_generator": "transformer",
                "max_queries_per_response": 2
            }
        )
        
        model = model_factory(config)
        
        # Forward pass with tree planning and retrieval
        with torch.no_grad():
            outputs = model(**small_batch)
            
        # Check output types
        assert outputs["tree_structure"].dtype == torch.long
        if mode == PrecisionMode.FP16:
            assert outputs["node_embeddings"].dtype == torch.float16
        else:
            assert outputs["node_embeddings"].dtype == torch.float32

@pytest.mark.precision
@pytest.mark.benchmark
def test_tree_planning_memory_usage(model_factory, small_batch, memory_tracker, device):
    """Benchmark memory usage of tree planning with different precisions"""
    results = {}
    
    for mode in [PrecisionMode.FP16, PrecisionMode.FP32]:
        memory_tracker.reset()
        
        config = ModelConfig(
            hidden_size=256,
            num_layers=4,
            precision=PrecisionConfig(mode=mode),
            tree_planner={
                "enabled": True,
                "num_tree_layers": 4,
                "tree_hidden_size": 128
            }
        )
        
        model = model_factory(config)
        
        # Measure memory during forward pass
        with torch.no_grad():
            _ = model(**small_batch)
            
        results[mode] = memory_tracker.get_max_memory()
        
        del model
        torch.cuda.empty_cache()
    
    # FP16 should use significantly less memory
    assert results[PrecisionMode.FP16] < results[PrecisionMode.FP32] * 0.6

@pytest.mark.precision
def test_tree_planning_numerical_stability(model_factory, small_batch, device):
    """Test numerical stability of tree planning across precision modes"""
    config = ModelConfig(
        hidden_size=128,
        num_layers=2,
        precision=PrecisionConfig(
            mode=PrecisionMode.FP16,
            mixed_precision=True,
            gradient_precision="fp32"
        ),
        tree_planner={
            "enabled": True,
            "num_tree_layers": 2,
            "tree_hidden_size": 64
        }
    )
    
    model = model_factory(config)
    optimizer = torch.optim.Adam(model.parameters())
    scaler = GradScaler(precision_config=config.precision)
    
    # Training step with mixed precision
    for _ in range(5):  # Multiple steps to check stability
        with torch.cuda.amp.autocast(enabled=True):
            outputs = model(**small_batch)
            loss = outputs["loss"]
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        
        # Check for NaN/inf values
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)
        
        # Check tree structure values
        assert not torch.isnan(outputs["tree_structure"]).any()
        assert not torch.isnan(outputs["node_embeddings"]).any()

@pytest.mark.precision
def test_tree_structure_generation(model_factory, device):
    """Test tree structure generation across precision modes"""
    input_text = "Explain the concept of quantum computing."
    
    for mode in [PrecisionMode.FP16, PrecisionMode.FP32]:
        config = ModelConfig(
            hidden_size=128,
            num_layers=2,
            precision=PrecisionConfig(mode=mode),
            tree_planner={
                "enabled": True,
                "num_tree_layers": 2,
                "tree_hidden_size": 64,
                "max_tree_depth": 3
            }
        )
        
        model = model_factory(config)
        
        # Generate tree structure
        with torch.no_grad():
            tree = model.tree_planner.generate_tree(input_text)
        
        # Validate tree structure
        assert tree.depth <= 3
        assert len(tree.get_all_nodes()) > 0
        
        # Check node embeddings precision
        for node in tree.get_all_nodes():
            if hasattr(node, 'embedding'):
                if mode == PrecisionMode.FP16:
                    assert node.embedding.dtype == torch.float16
                else:
                    assert node.embedding.dtype == torch.float32

@pytest.mark.precision
@pytest.mark.parametrize("batch_size", [1, 4, 8])
def test_batch_tree_planning(model_factory, batch_size, device):
    """Test tree planning with different batch sizes and precisions"""
    config = ModelConfig(
        hidden_size=128,
        num_layers=2,
        precision=PrecisionConfig(
            mode=PrecisionMode.FP16,
            mixed_precision=True
        ),
        tree_planner={
            "enabled": True,
            "num_tree_layers": 2,
            "tree_hidden_size": 64
        }
    )
    
    model = model_factory(config)
    
    # Create batch
    batch = {
        "input_ids": torch.randint(0, 1000, (batch_size, 16), device=device),
        "attention_mask": torch.ones(batch_size, 16, device=device)
    }
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**batch)
    
    # Check batch dimension
    assert outputs["tree_structure"].size(0) == batch_size
    assert outputs["node_embeddings"].size(0) == batch_size
