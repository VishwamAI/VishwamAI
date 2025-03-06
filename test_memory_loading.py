"""Test memory-efficient model loading for QwQ distillation."""
import os
import jax
import jax.numpy as jnp
import psutil
import time
from tqdm import tqdm

from vishwamai.model import VishwamAIModel, ModelConfig
from vishwamai.qwen_data import QwenDataLoader
from omegaconf import OmegaConf

def print_memory_usage():
    """Print current memory usage."""
    process = psutil.Process()
    memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
    print(f"Memory usage: {memory_gb:.2f} GB")

def test_chunk_sizes(qwq_path: str):
    """Test different chunk sizes for loading."""
    chunk_sizes = [16, 32, 64]
    results = []
    
    for chunk_size in chunk_sizes:
        print(f"\nTesting chunk size: {chunk_size}")
        start_mem = psutil.Process().memory_info().rss
        start_time = time.time()
        
        try:
            # Initialize loader
            loader = QwenDataLoader(
                safetensor_dir=qwq_path,
                batch_size=1,
                gradient_accumulation_steps=16,
                chunk_size=chunk_size
            )
            
            # Load first shard only for testing
            print("Loading first shard...")
            first_shard = next(loader.get_shard_stream())[1]
            
            end_time = time.time()
            peak_mem = psutil.Process().memory_info().rss
            mem_increase = (peak_mem - start_mem) / (1024 * 1024 * 1024)  # Convert to GB
            
            results.append({
                'chunk_size': chunk_size,
                'time': end_time - start_time,
                'memory_increase_gb': mem_increase,
                'success': True
            })
            
            print(f"Time taken: {end_time - start_time:.2f}s")
            print(f"Memory increase: {mem_increase:.2f}GB")
            
            # Clear memory
            del first_shard
            jax.clear_caches()
            
        except Exception as e:
            print(f"Failed with chunk size {chunk_size}: {str(e)}")
            results.append({
                'chunk_size': chunk_size,
                'error': str(e),
                'success': False
            })
    
    return results

def verify_model_loading():
    """Verify full model loading process."""
    print("\nVerifying model loading...")
    
    # Load configuration
    config = OmegaConf.load('configs/distillation_config.yaml')
    
    # Initialize model and loader
    model = VishwamAIModel(ModelConfig(**config.distillation.teacher_model.config))
    loader = QwenDataLoader(
        safetensor_dir="path/to/qwq",  # Update with actual path
        batch_size=1,
        gradient_accumulation_steps=16,
        chunk_size=32  # Use best chunk size from previous test
    )
    
    print("\nInitial memory usage:")
    print_memory_usage()
    
    try:
        print("\nLoading model weights in chunks...")
        params = loader.load_all_shards()
        model = model.bind({'params': params})
        
        print("\nPeak memory usage:")
        print_memory_usage()
        
        print("\nTesting model forward pass...")
        test_input = jnp.ones((1, 512), dtype=jnp.int32)
        _ = model(test_input)
        
        print("Model loading and forward pass successful!")
        return True
    except Exception as e:
        print(f"Model verification failed: {str(e)}")
        return False

def main():
    """Run memory loading tests."""
    print("Starting memory-efficient loading tests...")
    print(f"JAX devices: {jax.devices()}")
    print(f"Available memory: {psutil.virtual_memory().available / (1024**3):.2f}GB")
    
    # Update with actual QwQ path
    QWQ_PATH = os.environ.get("QWQ_PATH", "path/to/qwq")
    if not os.path.exists(QWQ_PATH):
        print(f"Please set QWQ_PATH environment variable to point to QwQ model directory")
        return 1
    
    # Test different chunk sizes
    results = test_chunk_sizes(QWQ_PATH)
    
    print("\nResults summary:")
    print("-" * 50)
    for result in results:
        if result['success']:
            print(f"Chunk size {result['chunk_size']}:")
            print(f"  Time: {result['time']:.2f}s")
            print(f"  Memory increase: {result['memory_increase_gb']:.2f}GB")
        else:
            print(f"Chunk size {result['chunk_size']}: Failed - {result['error']}")
    
    # Verify full model loading
    if verify_model_loading():
        print("\nAll tests passed!")
        return 0
    else:
        print("\nTests failed!")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())
