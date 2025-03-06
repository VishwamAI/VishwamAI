"""Optimized data loading for Qwen model distillation."""
import os
import jax
import jax.numpy as jnp
import safetensors.flax as stf
from typing import Dict, Iterator, Tuple, Optional
from functools import partial

class QwenDataLoader:
    """Efficient data loader for Qwen model training."""
    
    def __init__(
        self,
        safetensor_dir: str,
        batch_size: int = 4,
        max_sequence_length: int = 2048,
        dtype: str = "bfloat16"
    ):
        """Initialize Qwen data loader.
        
        Args:
            safetensor_dir: Directory containing Qwen safetensor files
            batch_size: Batch size per device
            max_sequence_length: Maximum sequence length
            dtype: Data type for tensors (default: bfloat16 for TPU)
        """
        self.safetensor_dir = safetensor_dir
        self.batch_size = batch_size
        self.max_sequence_length = max_sequence_length
        self.dtype = getattr(jnp, dtype)
        
        # Validate safetensor files
        self.shard_files = [
            f for f in os.listdir(safetensor_dir) 
            if f.endswith('.safetensors')
        ]
        assert len(self.shard_files) == 14, f"Expected 14 safetensor files, found {len(self.shard_files)}"
        
        # Sort shards for consistent loading
        self.shard_files.sort()
        
        # TPU device setup
        self.num_devices = jax.device_count()
        self.device_batch_size = batch_size // self.num_devices
        assert self.device_batch_size > 0, "Batch size must be >= number of devices"
    
    def _load_shard(self, shard_file: str) -> Dict:
        """Load a single safetensor shard with TPU optimization."""
        shard_path = os.path.join(self.safetensor_dir, shard_file)
        params = stf.load_file(shard_path)
        
        # Convert to desired dtype
        params = jax.tree_map(
            lambda x: x.astype(self.dtype) if x.dtype != jnp.int32 else x,
            params
        )
        
        return params

    @partial(jax.jit, static_argnums=(0,))
    def _prepare_batch(self, params: Dict) -> Dict:
        """Prepare batch for TPU training."""
        # Reshape for device sharding
        return jax.tree_map(
            lambda x: x.reshape((self.num_devices, -1) + x.shape[1:])
            if len(x.shape) > 1 else x,
            params
        )
    
    def load_all_shards(self) -> Dict:
        """Load all safetensor shards efficiently."""
        params = {}
        for shard_file in self.shard_files:
            shard_params = self._load_shard(shard_file)
            params.update(shard_params)
        return params
    
    def get_shard_stream(self) -> Iterator[Tuple[str, Dict]]:
        """Stream shards for memory-efficient processing."""
        for shard_file in self.shard_files:
            yield shard_file, self._load_shard(shard_file)
    
    def create_training_batch(
        self,
        input_ids: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None
    ) -> Dict:
        """Create training batch with TPU sharding."""
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask if attention_mask is not None 
            else jnp.ones_like(input_ids)
        }
        
        # Prepare for TPU
        return self._prepare_batch(batch)

# Usage example
if __name__ == "__main__":
    # Example usage
    loader = QwenDataLoader(
        safetensor_dir="path/to/qwen/safetensors",
        batch_size=8
    )
    
    # Load all shards
    params = loader.load_all_shards()
    print(f"Loaded {len(params)} parameters from 14 shards")
    
    # Or stream shards
    for shard_name, shard_params in loader.get_shard_stream():
        print(f"Processing shard: {shard_name}")
        # Process shard parameters...
