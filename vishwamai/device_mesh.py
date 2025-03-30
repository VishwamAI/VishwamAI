"""Device mesh management for TPU training."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils
from typing import Any, Dict, Optional, Tuple, List

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
        self.devices = jax.devices()
        self.num_devices = len(self.devices)
        
        # Create mesh shape and device mesh
        self.mesh_shape = self._compute_mesh_shape()
        self.mesh = self._create_device_mesh()
        self.sharding_rules = self._create_sharding_rules()
    
    def __enter__(self):
        """Enter mesh context."""
        return self.mesh.__enter__()
    
    def __exit__(self, exc_type, exc_value, traceback):
        """Exit mesh context."""
        return self.mesh.__exit__(exc_type, exc_value, traceback)

    def _compute_mesh_shape(self) -> Tuple[int, ...]:
        """Compute optimal device mesh shape."""
        if self.pipeline_parallel:
            return (2, 2, 2)  # (data, model, pipe)
        elif self.model_parallel and self.data_parallel:
            return (4, 2)  # (data, model)
        else:
            return (self.num_devices,)  # (data,)
    
    def _create_device_mesh(self) -> Mesh:
        """Create TPU device mesh with optimal topology mapping."""
        devices = np.array(self.devices).reshape(self.mesh_shape)
        if self.pipeline_parallel:
            return Mesh(devices, ("data", "model", "pipe"))
        elif self.model_parallel and self.data_parallel:
            return Mesh(devices, ("data", "model"))
        else:
            return Mesh(devices, ("data",))
    
    def _create_sharding_rules(self) -> Dict[str, Any]:
        """Create sharding rules for different tensor types."""
        if self.pipeline_parallel:
            return {
                "params": P("model", None),
                "optimizer_state": P(None),
                "attention": {
                    "query": P("data", None),
                    "key": P("data", None),
                    "value": P("data", None),
                    "output": P("data", None)
                }
            }
        elif self.model_parallel:
            return {
                "params": P("model", None),
                "optimizer_state": P(None),
                "attention": {
                    "query": P("data", None),
                    "key": P("data", None),
                    "value": P("data", None),
                    "output": P("data", None)
                }
            }
        else:
            return {
                "params": P(None),
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