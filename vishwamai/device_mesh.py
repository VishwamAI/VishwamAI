"""TPU device mesh and parallel processing utilities"""

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh
from jax.sharding import Sharding
from jax.sharding import PartitionSpec as P
from typing import Dict, Any, Optional, Tuple, List, Union
from contextlib import contextmanager
import numpy as np

class TPUMeshContext:
    """Manages TPU device mesh and data/model parallelism."""
    
    def __init__(
        self,
        config: Dict[str, Any],
        data_parallel: bool = True,
        model_parallel: bool = True,
        pipeline_parallel: bool = False
    ):
        self.config = config
        self.data_parallel = data_parallel
        self.model_parallel = model_parallel
        self.pipeline_parallel = pipeline_parallel
        
        # Get available devices
        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        
        # Determine mesh shape based on parallelism strategy
        self.mesh_shape = self._compute_mesh_shape()
        
        # Create device mesh
        self.device_mesh = self._create_device_mesh()
        
        # Create sharding rules
        self.sharding_rules = self._create_sharding_rules()
    
    def _compute_mesh_shape(self) -> Tuple[int, ...]:
        """Compute optimal device mesh shape."""
        if self.pipeline_parallel:
            # 2x2x2 topology for pipeline parallelism
            return (2, 2, 2)  # (data, model, pipe)
        elif self.model_parallel and self.data_parallel:
            # 4x2 topology for data and model parallelism
            return (4, 2)  # (data, model)
        else:
            # Use all devices for data parallelism
            return (self.num_devices,)
    
    def _create_device_mesh(self) -> Mesh:
        """Create TPU device mesh with optimal topology mapping."""
        if self.pipeline_parallel:
            # For 2x2x2 topology, create physical device mapping
            devices = np.array(self.devices).reshape(2, 2, 2)  # Reshape to match physical topology
            mesh = jax.sharding.Mesh(devices, ("data", "model", "pipe"))
        elif self.model_parallel and self.data_parallel:
            # For 4x2 topology (data parallel x model parallel)
            devices = np.array(self.devices).reshape(4, 2)
            mesh = jax.sharding.Mesh(devices, ("data", "model"))
        else:
            # Single dimension for pure data parallelism
            devices = np.array(self.devices).reshape(-1)
            mesh = jax.sharding.Mesh(devices, ("data",))
        
        return mesh
    
    def _create_sharding_rules(self) -> Dict[str, Any]:
        """Create sharding rules for different tensor types."""
        if self.pipeline_parallel:
            # Optimal sharding for 2x2x2 topology
            return {
                "weights": P("model", "pipe", None),  # Shard across model and pipeline dimensions
                "biases": P("model", "pipe"),
                "activations": P("data", None, "pipe"),  # Shard activations across data parallel and pipeline dimensions
                "gradients": P("data", "model", "pipe"),  # Shard gradients across all dimensions
                "optimizer_state": P("model", "pipe", None),
                "attention": {
                    "query": P("data", "model", None),
                    "key": P("data", "model", None),
                    "value": P("data", "model", None),
                    "output": P("data", "model", None)
                }
            }
        elif self.model_parallel and self.data_parallel:
            # 4x2 topology for data and model parallelism
            return {
                "weights": P("model", None),
                "biases": P("model"),
                "activations": P("data", None),
                "gradients": P("data", "model"),
                "optimizer_state": P("model", None),
                "attention": {
                    "query": P("data", None),
                    "key": P("data", None),
                    "value": P("data", None),
                    "output": P("data", None)
                }
            }
        else:
            # Simple data parallel only
            return {
                "weights": P(None),
                "biases": P(None),
                "activations": P("data", None),
                "gradients": P("data", None),
                "optimizer_state": P(None),
                "attention": {
                    "query": P("data", None),
                    "key": P("data", None),
                    "value": P("data", None),
                    "output": P("data", None)
                }
            }
    
    def shard_params(
        self,
        params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Shard model parameters across devices."""
        def _shard(param, param_type="weights"):
            return jax.device_put_sharded(
                param,
                self.devices,
                sharding=self.sharding_rules[param_type]
            )
        
        return jax.tree_map(_shard, params)
    
    def shard_batch(
        self,
        batch: Dict[str, jnp.ndarray]
    ) -> Dict[str, jnp.ndarray]:
        """Shard batch data across devices."""
        def _shard_data(x):
            return jax.device_put_sharded(
                x,
                self.devices,
                sharding=P("data", None)
            )
        
        return jax.tree_map(_shard_data, batch)
    
    def all_reduce(
        self,
        x: jnp.ndarray,
        reduce_type: str = "sum"
    ) -> jnp.ndarray:
        """Perform all-reduce across devices."""
        if reduce_type == "sum":
            return jax.lax.psum(x, "data")
        elif reduce_type == "mean":
            return jax.lax.pmean(x, "data")
        else:
            raise ValueError(f"Unsupported reduce type: {reduce_type}")
    
    def replicate(
        self,
        x: Any
    ) -> Any:
        """Replicate data across devices."""
        return jax.device_put_replicated(x, self.devices)
    
    def optimize_layout(
        self,
        x: jnp.ndarray,
        layout_type: str = "matmul"
    ) -> jnp.ndarray:
        """Optimize tensor layout for TPU operations."""
        if layout_type == "matmul":
            # Optimize for matrix multiplication
            if x.ndim == 4:  # BHQK format
                return x.transpose((0, 2, 1, 3))  # -> BQHK
            elif x.ndim == 3:  # BLD format
                return x.transpose((1, 0, 2))  # -> LBD
        elif layout_type == "attention":
            # Optimize for attention operations
            if x.ndim == 4:  # BSNH format
                return x.transpose((0, 2, 1, 3))  # -> BNSH
        
        return x
    
    def create_pipeline_schedule(
        self,
        num_micro_batches: int
    ) -> List[Dict[str, Any]]:
        """Create pipeline execution schedule."""
        if not self.pipeline_parallel:
            raise ValueError("Pipeline parallelism not enabled")
        
        num_stages = self.mesh_shape[-1]
        schedule = []
        
        # Forward passes
        for micro_batch in range(num_micro_batches):
            for stage in range(num_stages):
                schedule.append({
                    "type": "forward",
                    "micro_batch": micro_batch,
                    "stage": stage
                })
        
        # Backward passes
        for micro_batch in reversed(range(num_micro_batches)):
            for stage in reversed(range(num_stages)):
                schedule.append({
                    "type": "backward",
                    "micro_batch": micro_batch,
                    "stage": stage
                })
        
        return schedule
    
    @contextmanager
    def mesh_context(self):
        """Context manager for device mesh."""
        with self.device_mesh:
            yield
    
    def get_mesh_shape(self) -> Tuple[int, ...]:
        """Get device mesh shape."""
        return self.mesh_shape
    
    def get_devices(self) -> List[Any]:
        """Get list of available devices."""
        return self.devices
    
    def get_sharding_rules(self) -> Dict[str, Any]:
        """Get sharding rules."""
        return self.sharding_rules
    
    def merge_gradients(
        self,
        grads: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Merge gradients across devices."""
        return jax.tree_map(
            lambda g: self.all_reduce(g, "mean") if g is not None else None,
            grads
        )
    
    def create_train_state(
        self,
        state: Any,
        is_training: bool = True
    ) -> Any:
        """Create training state with appropriate sharding."""
        if is_training:
            # Shard parameters and optimizer state
            state = state.replace(
                params=self.shard_params(state.params),
                opt_state=jax.tree_map(
                    lambda x: self.shard_params(x, "optimizer_state"),
                    state.opt_state
                )
            )
        else:
            # Only shard parameters for inference
            state = state.replace(
                params=self.shard_params(state.params)
            )
        
        return state
    
    def get_optimal_batch_size(self) -> int:
        """Calculate optimal batch size for current device configuration."""
        memory_per_device = self.config["tpu"].get("memory_per_device", 16)  # GB
        model_size = self.config["model"].get("hidden_dim", 768) ** 2 * 4  # Bytes
        
        max_batch_size = (memory_per_device * 1e9 * 0.7) // model_size
        max_batch_size = int(max_batch_size)
        
        # Round down to nearest multiple of number of data parallel devices
        data_parallel_size = self.mesh_shape[0]
        return (max_batch_size // data_parallel_size) * data_parallel_size
    
    def reshape_for_padding(
        self,
        x: jnp.ndarray,
        block_size: int = 128
    ) -> Tuple[jnp.ndarray, Dict[str, Any]]:
        """Reshape and pad tensor for TPU operations."""
        shape = x.shape
        
        # Calculate padding
        pad_size = [
            (0, (block_size - dim % block_size) % block_size)
            for dim in shape
        ]
        
        # Pad tensor
        x_padded = jnp.pad(
            x,
            pad_size,
            mode='constant',
            constant_values=0
        )
        
        metadata = {
            "original_shape": shape,
            "pad_size": pad_size
        }
        
        return x_padded, metadata
    
    def remove_padding(
        self,
        x: jnp.ndarray,
        metadata: Dict[str, Any]
    ) -> jnp.ndarray:
        """Remove padding from tensor."""
        original_shape = metadata["original_shape"]
        return x[tuple(slice(0, dim) for dim in original_shape)]

def create_device_mesh(
    config: Dict[str, Any]
) -> TPUMeshContext:
    """Create TPU device mesh context."""
    return TPUMeshContext(
        config=config,
        data_parallel=config["training"].get("data_parallel", True),
        model_parallel=config["training"].get("model_parallel", True),
        pipeline_parallel=config["training"].get("pipeline_parallel", False)
    )