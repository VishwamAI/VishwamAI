"""Optimized data loading for Qwen model distillation."""
import os
import jax
import jax.numpy as jnp
import safetensors
from safetensors.flax import save_file, load_file
import numpy as np
from typing import Dict, Iterator, Tuple, Optional
from functools import partial
import gc

from .tensor_utils import (
    load_tensor_chunks, 
    get_memory_usage,
    suggest_chunk_size,
    TENSOR_CACHE
)

class QwenDataLoader:
    """Efficient data loader for Qwen model training with memory optimization."""
    
    def __init__(
        self,
        safetensor_dir: str,
        batch_size: int = 4,
        max_sequence_length: int = 2048,
        dtype: str = "bfloat16",
        gradient_accumulation_steps: int = 1,
        chunk_size: Optional[int] = None,  # Auto-determine if None
        target_chunk_gb: float = 0.5
    ):
        """Initialize Qwen data loader.
        
        Args:
            safetensor_dir: Directory containing Qwen safetensor files
            batch_size: Global batch size
            max_sequence_length: Maximum sequence length
            dtype: Data type for tensors (default: bfloat16 for TPU)
            gradient_accumulation_steps: Number of gradient accumulation steps
            chunk_size: Size of chunks for memory-efficient loading
            target_chunk_gb: Target size per chunk in GB if chunk_size is None
        """
        self.safetensor_dir = safetensor_dir
        self.max_sequence_length = max_sequence_length
        self.dtype = getattr(jnp, dtype)
        self.target_chunk_gb = target_chunk_gb
        
        # TPU device setup
        self.num_devices = jax.device_count()
        
        # Adjust batch size for gradient accumulation
        self.global_batch_size = batch_size * gradient_accumulation_steps
        self.per_device_batch_size = max(1, self.global_batch_size // self.num_devices)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Validate safetensor files
        self.shard_files = sorted([
            f for f in os.listdir(safetensor_dir) 
            if f.endswith('.safetensors')
        ])
        if len(self.shard_files) != 14:
            raise ValueError(f"Expected 14 safetensor files, found {len(self.shard_files)}")
        
        # Determine chunk size if not provided
        if chunk_size is None:
            # Get a sample tensor shape from first shard
            first_shard = os.path.join(safetensor_dir, self.shard_files[0])
            with safetensors.safe_open(first_shard, framework="flax") as f:
                sample_key = next(iter(f.keys()))
                shape = f.get_tensor_info(sample_key)['shape']
                self.chunk_size = suggest_chunk_size(shape, self.dtype, target_chunk_gb)
        else:
            self.chunk_size = chunk_size
        
        print(f"Initialized loader with:"
              f"\n - {self.num_devices} devices"
              f"\n - Global batch size: {self.global_batch_size}"
              f"\n - Per-device batch size: {self.per_device_batch_size}"
              f"\n - Gradient accumulation steps: {gradient_accumulation_steps}"
              f"\n - Chunk size: {self.chunk_size}"
              f"\n - Initial memory usage: {get_memory_usage():.2f}GB")
    
    def _load_shard_in_chunks(self, shard_file: str) -> Dict:
        """Load a single safetensor shard in memory-efficient chunks."""
        shard_path = os.path.join(self.safetensor_dir, shard_file)
        
        try:
            # Check cache first
            cached_params = TENSOR_CACHE.get(shard_file)
            if cached_params is not None:
                return cached_params
            
            params = {}
            with safetensors.safe_open(shard_path, framework="flax") as f:
                for key in f.keys():
                    # Load tensor in chunks
                    tensor = load_tensor_chunks(
                        shard_path,
                        key,
                        chunk_size=self.chunk_size,
                        dtype=self.dtype
                    )
                    params[key] = tensor
            
            # Cache the loaded parameters
            TENSOR_CACHE.add(shard_file, params)
            return params
            
        except Exception as e:
            print(f"Error loading shard {shard_file}: {str(e)}")
            print(f"Memory usage at error: {get_memory_usage():.2f}GB")
            raise

    def load_all_shards(self) -> Dict:
        """Load all safetensor shards with memory management."""
        params = {}
        initial_memory = get_memory_usage()
        
        try:
            for shard_file in self.shard_files:
                print(f"Loading shard {shard_file}...")
                print(f"Memory before: {get_memory_usage():.2f}GB")
                
                shard_params = self._load_shard_in_chunks(shard_file)
                params.update(shard_params)
                
                # Clear memory after each shard
                jax.clear_caches()
                gc.collect()
                
                print(f"Memory after: {get_memory_usage():.2f}GB")
                print(f"Memory increase: {get_memory_usage() - initial_memory:.2f}GB")
        
        except Exception as e:
            print(f"Error in load_all_shards: {str(e)}")
            print(f"Final memory usage: {get_memory_usage():.2f}GB")
            raise
            
        return params
    
    def get_shard_stream(self) -> Iterator[Tuple[str, Dict]]:
        """Stream shards with memory cleanup between shards."""
        for shard_file in self.shard_files:
            yield shard_file, self._load_shard_in_chunks(shard_file)
            
            # Clear memory after each shard
            jax.clear_caches()
            gc.collect()
    
    @partial(jax.jit, static_argnums=(0,))
    def _prepare_batch(self, params: Dict) -> Dict:
        """Prepare batch for TPU training."""
        def reshape_for_devices(x):
            if len(x.shape) > 1:
                return x.reshape((self.num_devices, self.per_device_batch_size) + x.shape[1:])
            return x
            
        return jax.tree_map(reshape_for_devices, params)
    
    def create_training_batch(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        labels: Optional[jnp.ndarray] = None
    ) -> Dict:
        """Create training batch with TPU sharding and gradient accumulation."""
        # Handle smaller batches with padding
        if input_ids.shape[0] < self.global_batch_size:
            pad_amount = self.global_batch_size - input_ids.shape[0]
            input_ids = jnp.pad(input_ids, ((0, pad_amount), (0, 0)))
            if attention_mask is not None:
                attention_mask = jnp.pad(attention_mask, ((0, pad_amount), (0, 0)))
            if labels is not None:
                labels = jnp.pad(labels, ((0, pad_amount), (0, 0)))
        
        # Create and split batch
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask if attention_mask is not None 
            else jnp.ones_like(input_ids)
        }
        if labels is not None:
            batch['labels'] = labels
        
        # Split across devices
        return self._prepare_batch(batch)

# Example usage
if __name__ == "__main__":
    # Test loader with memory monitoring
    print(f"Initial memory: {get_memory_usage():.2f}GB")
    
    loader = QwenDataLoader(
        safetensor_dir="path/to/qwen/safetensors",
        batch_size=4,
        gradient_accumulation_steps=4
    )
    
    # Test batch creation
    test_input = jnp.ones((8, 512), dtype=jnp.int32)
    batch = loader.create_training_batch(test_input)
    print(f"Final memory: {get_memory_usage():.2f}GB")
    print("Test batch shape:", batch['input_ids'].shape)
