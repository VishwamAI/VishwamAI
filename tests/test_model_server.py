import pytest
import torch
import json
import httpx
from fastapi.testclient import TestClient
from vishwamai.serve import app, load_model
from vishwamai.model import Transformer, ModelArgs

# Initialize test client
client = TestClient(app)

@pytest.fixture(scope="module")
def model_components():
    """Fixture to provide model components for testing."""
    try:
        model, cache_module, memory_module, tree_module = load_model()
        return {
            "model": model,
            "cache": cache_module,
            "memory": memory_module,
            "tree": tree_module
        }
    except Exception as e:
        pytest.skip(f"Model loading failed: {str(e)}")

def test_health_check():
    """Test health check endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_model_config():
    """Test model configuration endpoint."""
    response = client.get("/config")
    assert response.status_code == 200
    config = response.json()
    assert "model_type" in config
    assert "model_size" in config
    assert "version" in config
    assert "components" in config

@pytest.mark.parametrize("input_text", [
    "What is deep learning?",
    "Write a Python function to sort a list.",
    "Explain quantum computing."
])
def test_model_prediction(model_components, input_text):
    """Test model prediction endpoint with various inputs."""
    response = client.post(
        "/predict",
        json={
            "text": input_text,
            "max_length": 100,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 50,
            "use_cache": True
        }
    )
    
    assert response.status_code == 200
    result = response.json()
    assert "generated_text" in result
    assert "performance_metrics" in result
    assert len(result["generated_text"]) > 0
    assert result["performance_metrics"]["total_tokens"] > 0

def test_model_components(model_components):
    """Test individual model components."""
    model = model_components["model"]
    cache = model_components["cache"]
    memory = model_components["memory"]
    tree = model_components["tree"]
    
    # Test model forward pass
    test_input = torch.randint(0, 32000, (1, 32)).cuda()
    with torch.inference_mode():
        output = model(test_input)
        assert output.shape[0] == 1
        
        # Test cache module
        cache_output = cache(output)
        assert cache_output.shape == output.shape
        
        # Test memory module
        memory_output = memory(cache_output)
        assert memory_output.shape == output.shape
        
        # Test tree module
        tree_output = tree(memory_output)
        assert tree_output.shape == output.shape

def test_error_handling():
    """Test error handling for invalid inputs."""
    # Test missing text
    response = client.post("/predict", json={})
    assert response.status_code == 422
    
    # Test invalid temperature
    response = client.post(
        "/predict",
        json={
            "text": "Test",
            "temperature": 2.0  # Invalid temperature > 1
        }
    )
    assert response.status_code == 422
    
    # Test invalid max_length
    response = client.post(
        "/predict",
        json={
            "text": "Test",
            "max_length": -1  # Invalid negative length
        }
    )
    assert response.status_code == 422

@pytest.mark.asyncio
async def test_concurrent_requests():
    """Test handling of concurrent requests."""
    async with httpx.AsyncClient(app=app, base_url="http://test") as ac:
        # Make multiple concurrent requests
        tasks = []
        for _ in range(5):
            task = ac.post(
                "/predict",
                json={
                    "text": "Test concurrent requests",
                    "max_length": 50
                }
            )
            tasks.append(task)
        
        # Wait for all requests to complete
        responses = await asyncio.gather(*tasks)
        
        # Verify all responses
        for response in responses:
            assert response.status_code == 200
            assert "generated_text" in response.json()

def test_performance_metrics(model_components):
    """Test performance monitoring."""
    response = client.post(
        "/predict",
        json={
            "text": "Test performance metrics",
            "max_length": 100
        }
    )
    
    metrics = response.json()["performance_metrics"]
    assert "total_tokens" in metrics
    assert "generation_time" in metrics
    assert "memory_usage" in metrics
    assert metrics["memory_usage"] > 0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
