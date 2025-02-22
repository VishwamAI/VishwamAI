"""
Tests for information retrieval components with different precision modes
"""
import pytest
import torch
import numpy as np

from vishwamai.config import ModelConfig, PrecisionConfig, PrecisionMode
from vishwamai.model import VishwamaiModel
from vishwamai.training.optimization import GradScaler

@pytest.mark.precision
@pytest.mark.gpu
def test_query_generator_precision(model_factory, precision_test_cases, device):
    """Test query generator with different precision modes"""
    for test_case in precision_test_cases:
        config = ModelConfig(
            hidden_size=128,
            num_layers=2,
            precision=PrecisionConfig(
                mode=test_case["mode"],
                mixed_precision=test_case["mixed_precision"],
                gradient_precision=test_case["gradient_precision"]
            ),
            information_retrieval={
                "enabled": True,
                "search_query_generator": "transformer",
                "max_queries_per_response": 2
            }
        )
        
        model = model_factory(config)
        
        # Check query generator parameters precision
        query_generator = model.query_generator
        for param in query_generator.parameters():
            assert param.dtype == test_case["expected_dtype"]

@pytest.mark.precision
def test_retrieval_with_mixed_precision(model_factory, small_batch, device):
    """Test information retrieval with mixed precision"""
    config = ModelConfig(
        hidden_size=128,
        num_layers=2,
        precision=PrecisionConfig(
            mode=PrecisionMode.FP16,
            mixed_precision=True,
            gradient_precision="fp32"
        ),
        information_retrieval={
            "enabled": True,
            "search_query_generator": "transformer",
            "max_queries_per_response": 2
        }
    )
    
    model = model_factory(config)
    scaler = GradScaler(precision_config=config.precision)
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training step with mixed precision
    with torch.cuda.amp.autocast(enabled=True):
        outputs = model(**small_batch)
        loss = outputs["loss"]
        queries = outputs["generated_queries"]
        
    # Check query tensor precision
    assert queries.dtype == torch.float16
    
    # Backward pass
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    
    # Check gradient precision
    for param in model.query_generator.parameters():
        if param.grad is not None:
            assert param.grad.dtype == torch.float32

@pytest.mark.precision
@pytest.mark.benchmark
def test_retrieval_memory_efficiency(model_factory, memory_tracker, device):
    """Test memory efficiency of retrieval across precision modes"""
    input_texts = ["Explain quantum computing", "Describe neural networks"]
    results = {}
    
    for mode in [PrecisionMode.FP16, PrecisionMode.FP32]:
        memory_tracker.reset()
        
        config = ModelConfig(
            hidden_size=256,
            num_layers=4,
            precision=PrecisionConfig(mode=mode),
            information_retrieval={
                "enabled": True,
                "search_query_generator": "transformer",
                "max_queries_per_response": 3
            }
        )
        
        model = model_factory(config)
        
        # Generate queries and measure memory
        with torch.no_grad():
            queries = model.generate_search_queries(input_texts)
            results[mode] = memory_tracker.get_max_memory()
            
        del model, queries
        torch.cuda.empty_cache()
    
    # FP16 should use less memory
    assert results[PrecisionMode.FP16] < results[PrecisionMode.FP32] * 0.6

@pytest.mark.precision
def test_search_query_generation(model_factory, device):
    """Test search query generation with different precisions"""
    input_text = "How do quantum computers work?"
    
    for mode in [PrecisionMode.FP16, PrecisionMode.FP32]:
        config = ModelConfig(
            hidden_size=128,
            num_layers=2,
            precision=PrecisionConfig(mode=mode),
            information_retrieval={
                "enabled": True,
                "search_query_generator": "transformer",
                "max_queries_per_response": 2
            }
        )
        
        model = model_factory(config)
        
        # Generate search queries
        with torch.no_grad():
            queries = model.generate_search_queries([input_text])
            
        # Check query tensor properties
        assert len(queries) > 0
        if mode == PrecisionMode.FP16:
            assert queries[0].dtype == torch.float16
        else:
            assert queries[0].dtype == torch.float32

@pytest.mark.precision
def test_retrieval_integration(model_factory, small_batch, device):
    """Test integration of retrieval with tree planning"""
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
        },
        information_retrieval={
            "enabled": True,
            "search_query_generator": "transformer",
            "max_queries_per_response": 2
        }
    )
    
    model = model_factory(config)
    
    # Forward pass
    with torch.no_grad():
        outputs = model(**small_batch)
        
    # Check outputs
    assert "tree_structure" in outputs
    assert "generated_queries" in outputs
    assert "retrieved_context" in outputs
    
    # Check precision consistency
    assert outputs["generated_queries"].dtype == torch.float16
    assert outputs["retrieved_context"].dtype == torch.float16

@pytest.mark.precision
@pytest.mark.parametrize("max_queries", [1, 2, 3])
def test_multiple_query_generation(model_factory, max_queries, device):
    """Test generation of multiple search queries"""
    config = ModelConfig(
        hidden_size=128,
        num_layers=2,
        precision=PrecisionConfig(
            mode=PrecisionMode.FP16,
            mixed_precision=True
        ),
        information_retrieval={
            "enabled": True,
            "search_query_generator": "transformer",
            "max_queries_per_response": max_queries
        }
    )
    
    model = model_factory(config)
    input_text = "Explain the theory of relativity"
    
    # Generate queries
    with torch.no_grad():
        queries = model.generate_search_queries([input_text])
        
    # Check number of queries
    assert len(queries) <= max_queries
    
    # Check query properties
    for query in queries:
        assert query.dtype == torch.float16
        assert not torch.isnan(query).any()

@pytest.mark.precision
def test_retrieval_cache_precision(model_factory, device):
    """Test precision handling in retrieval cache"""
    config = ModelConfig(
        hidden_size=128,
        num_layers=2,
        precision=PrecisionConfig(
            mode=PrecisionMode.FP16,
            mixed_precision=True
        ),
        information_retrieval={
            "enabled": True,
            "search_query_generator": "transformer",
            "max_queries_per_response": 2,
            "cache_size": 100
        }
    )
    
    model = model_factory(config)
    
    # Test queries
    queries = [
        "What is quantum computing?",
        "How do neural networks work?"
    ]
    
    # Generate and cache results
    with torch.no_grad():
        for query in queries:
            result = model.retrieve_information(query)
            assert result.dtype == torch.float16
            
        # Check cached results
        cached_result = model.retrieve_information(queries[0])
        assert cached_result.dtype == torch.float16
