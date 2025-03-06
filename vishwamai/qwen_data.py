"""Optimized data loading for Qwen model distillation."""
import os
import jax
import jax.numpy as jnp
import safetensors.flax as stf
from typing import Dict, Iterator, Tuple, Optional
from functools import partial
import gc

class QwenDataLoader:
    """Efficient data loader for Qwen model training."""
    
    def __init__(
        self,
        safetensor_dir: str,
        batch_size: int = 4,
        max_sequence_length: int = 2048,
        dtype: str = "bfloat16",
        gradient_accumulation_steps: int = 1,
        chunk_size: int = 64  # Size of chunks for memory-efficient loading
    ):
        """Initialize Qwen data loader.
        
        Args:
            safetensor_dir: Directory containing Qwen safetensor files
            batch_size: Global batch size
            max_sequence_length: Maximum sequence length
            dtype: Data type for tensors (default: bfloat16 for TPU)
            gradient_accumulation_steps: Number of gradient accumulation steps
            chunk_size: Size of chunks for memory-efficient loading
        """
        self.safetensor_dir = safetensor_dir
        self.max_sequence_length = max_sequence_length
        self.dtype = getattr(jnp, dtype)
        self.chunk_size = chunk_size
        
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
        
        print(f"Initialized loader with:"
              f"\n - {self.num_devices} devices"
              f"\n - Global batch size: {self.global_batch_size}"
              f"\n - Per-device batch size: {self.per_device_batch_size}"
              f"\n - Gradient accumulation steps: {gradient_accumulation_steps}")
    
    def _load_shard_in_chunks(self, shard_file: str) -> Dict:
        """Load a single safetensor shard in chunks to manage memory."""
        shard_path = os.path.join(self.safetensor_dir, shard_file)
        
        # First, get tensor info without loading data
        tensor_info = stf.safe_open(shard_path, framework="flax").info()
        
        params = {}
        for key, info in tensor_info.items():
            shape = info['shape']
            
            # If tensor is large, load in chunks
            if shape[0] > self.chunk_size:
                chunks = []
                for i in range(0, shape[0], self.chunk_size):
                    end = min(i + self.chunk_size, shape[0])
                    chunk = stf.load_file(
                        shard_path,
                        tensor_info={key: {'slice': [i, end]}}
                    )[key]
                    chunks.append(chunk)
                    # Clear memory after each chunk
                    jax.clear_caches()
                    gc.collect()
                
                # Concatenate chunks
                params[key] = jnp.concatenate(chunks, axis=0)
            else:
                # Load small tensors directly
                params[key] = stf.load_file(shard_path)[key]
            
            # Convert to desired dtype
            if params[key].dtype != jnp.int32:
                params[key] = params[key].astype(self.dtype)
        
        return params

    def load_all_shards(self) -> Dict:
        """Load all safetensor shards efficiently."""
        params = {}
        for shard_file in self.shard_files:
            print(f"Loading shard {shard_file}...")
            shard_params = self._load_shard_in_chunks(shard_file)
            params.update(shard_params)
            # Clear memory after each shard
            jax.clear_caches()
            gc.collect()
        return params
    
    def get_shard_stream(self) -> Iterator[Tuple[str, Dict]]:
        """Stream shards for memory-efficient processing."""
        for shard_file in self.shard_files:
            yield shard_file, self._load_shard_in_chunks(shard_file)
            # Clear memory after processing each shard
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
    # Test loader
    loader = QwenDataLoader(
        safetensor_dir="path/to/qwen/safetensors",
        batch_size=4,
        gradient_accumulation_steps=4,
        chunk_size=32
    )
    
    # Test batch creation
    test_input = jnp.ones((8, 512), dtype=jnp.int32)
    batch = loader.create_training_batch(test_input)
    print("Test batch shape:", batch['input_ids'].shape)
