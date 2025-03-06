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
        dtype: str = "bfloat16",
        gradient_accumulation_steps: int = 1
    ):
        """Initialize Qwen data loader.
        
        Args:
            safetensor_dir: Directory containing Qwen safetensor files
            batch_size: Global batch size
            max_sequence_length: Maximum sequence length
            dtype: Data type for tensors (default: bfloat16 for TPU)
            gradient_accumulation_steps: Number of gradient accumulation steps
        """
        self.safetensor_dir = safetensor_dir
        self.max_sequence_length = max_sequence_length
        self.dtype = getattr(jnp, dtype)
        
        # TPU device setup
        self.num_devices = jax.device_count()
        
        # Adjust batch size for gradient accumulation
        self.global_batch_size = batch_size * gradient_accumulation_steps
        self.per_device_batch_size = max(1, self.global_batch_size // self.num_devices)
        self.gradient_accumulation_steps = gradient_accumulation_steps
        
        # Validate safetensor files
        self.shard_files = [
            f for f in os.listdir(safetensor_dir) 
            if f.endswith('.safetensors')
        ]
        if len(self.shard_files) != 14:
            raise ValueError(f"Expected 14 safetensor files, found {len(self.shard_files)}")
        
        # Sort shards for consistent loading
        self.shard_files.sort()
        
        print(f"Initialized loader with:"
              f"\n - {self.num_devices} devices"
              f"\n - Global batch size: {self.global_batch_size}"
              f"\n - Per-device batch size: {self.per_device_batch_size}"
              f"\n - Gradient accumulation steps: {gradient_accumulation_steps}")
    
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
        # Reshape for device sharding and gradient accumulation
        def reshape_for_devices(x):
            if len(x.shape) > 1:
                # Reshape: (batch_size, ...) -> (num_devices, per_device_batch_size, ...)
                return x.reshape((self.num_devices, self.per_device_batch_size) + x.shape[1:])
            return x
            
        return jax.tree_map(reshape_for_devices, params)
    
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
        attention_mask: Optional[jnp.ndarray] = None,
        labels: Optional[jnp.ndarray] = None
    ) -> Dict:
        """Create training batch with TPU sharding and gradient accumulation."""
        # Ensure inputs are the right shape
        if input_ids.shape[0] < self.global_batch_size:
            # Pad batch if needed
            pad_amount = self.global_batch_size - input_ids.shape[0]
            input_ids = jnp.pad(input_ids, ((0, pad_amount), (0, 0)))
            if attention_mask is not None:
                attention_mask = jnp.pad(attention_mask, ((0, pad_amount), (0, 0)))
            if labels is not None:
                labels = jnp.pad(labels, ((0, pad_amount), (0, 0)))
        
        batch = {
            'input_ids': input_ids,
            'attention_mask': attention_mask if attention_mask is not None 
            else jnp.ones_like(input_ids)
        }
        if labels is not None:
            batch['labels'] = labels
        
        # Prepare for TPU
        return self._prepare_batch(batch)

# Example usage
if __name__ == "__main__":
    # Test loader
    loader = QwenDataLoader(
        safetensor_dir="path/to/qwen/safetensors",
        batch_size=4,
        gradient_accumulation_steps=4  # Effective batch size = 16
    )
    
    # Load test batch
    test_input = jnp.ones((8, 512), dtype=jnp.int32)
    batch = loader.create_training_batch(test_input)
    print("Test batch shape:", batch['input_ids'].shape)
