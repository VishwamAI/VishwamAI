"""Device mesh management for TPU training."""

import jax
import jax.numpy as jnp
import numpy as np
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P
from jax.experimental import mesh_utils
from typing import Any, Dict, Optional, Tuple, List
import time

class TPUMeshContext:
    """Context manager for TPU mesh execution."""
    
    def __init__(self, mesh_config: Dict[str, Any], data_parallel: bool = True):
        if 'tpu' not in mesh_config or 'mesh_shape' not in mesh_config['tpu']:
            raise ValueError("mesh_config must contain 'tpu.mesh_shape'")
            
        mesh_shape = mesh_config['tpu']['mesh_shape']
        if not isinstance(mesh_shape, (list, tuple)) or any(x <= 0 for x in mesh_shape):
            raise ValueError("mesh_shape must be positive")
            
        self.mesh_shape = mesh_shape
        self.data_parallel = data_parallel
        self.devices = jax.devices()
        self.mesh = None
        
    def __enter__(self):
        if self.data_parallel:
            self.mesh = jax.sharding.Mesh(self.devices, ('data',))
        else:
            self.mesh = jax.sharding.Mesh(
                mesh_utils.create_device_mesh(self.mesh_shape),
                ('batch', 'model')
            )
        return self.mesh.__enter__()
        
    def __exit__(self, exc_type, exc_value, traceback):
        if self.mesh:
            return self.mesh.__exit__(exc_type, exc_value, traceback)

    def _compute_mesh_shape(self) -> Tuple[int, ...]:
        """Compute cost-efficient device mesh shape based on workload."""
        if self.pipeline_parallel:
            # For very large models, use 3D parallelism
            if self.num_devices >= 8:
                # Cost-efficient pipeline depth
                pipe_depth = min(2, self.config.get("num_layers", 32) // 16)
                model_parallel = min(2, self.num_devices // (2 * pipe_depth))
                data_parallel = self.num_devices // (pipe_depth * model_parallel)
                return (data_parallel, model_parallel, pipe_depth)
            return (2, 2, 2)
            
        elif self.model_parallel and self.data_parallel:
            # For medium models, prefer data parallelism
            if self.num_devices >= 4:
                # Determine if model parallelism is worth the communication cost
                hidden_dim = self.config.get("hidden_dim", 2048)
                if hidden_dim >= 8192:  # Only use model parallel for very large models
                    model_parallel = min(2, self.num_devices // 4)
                    return (self.num_devices // model_parallel, model_parallel)
                else:
                    return (self.num_devices, 1)  # Pure data parallel
            return (2, 2)
            
        # Default to pure data parallelism for cost efficiency
        return (self.num_devices,)

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
                # Use 128x128 tiling for TPU MXU efficiency
                B, H, Q, K = x.shape
                tile_size = 128
                
                # Pad dimensions to tile size
                pad_Q = (tile_size - Q % tile_size) % tile_size
                pad_K = (tile_size - K % tile_size) % tile_size
                
                if pad_Q > 0 or pad_K > 0:
                    x = jnp.pad(x, ((0,0), (0,0), (0,pad_Q), (0,pad_K)))
                
                # Reshape to expose tile structure
                Q_tiles = (Q + pad_Q) // tile_size
                K_tiles = (K + pad_K) // tile_size
                x = x.reshape(B, H, Q_tiles, tile_size, K_tiles, tile_size)
                
                # Reorder for memory locality
                return x.transpose(0, 2, 4, 1, 3, 5)  # B,Qt,Kt,H,Qs,Ks
                
            elif x.ndim == 3:  # BLD format
                # Optimize for typical encoder/decoder layouts
                B, L, D = x.shape
                tile_size = 128
                
                # Pad sequence and hidden dimensions
                pad_L = (tile_size - L % tile_size) % tile_size
                pad_D = (tile_size - D % tile_size) % tile_size
                
                if pad_L > 0 or pad_D > 0:
                    x = jnp.pad(x, ((0,0), (0,pad_L), (0,pad_D)))
                
                # Reshape and transpose for TPU efficiency
                L_tiles = (L + pad_L) // tile_size
                D_tiles = (D + pad_D) // tile_size
                x = x.reshape(B, L_tiles, tile_size, D_tiles, tile_size)
                return x.transpose(0, 1, 3, 2, 4)  # B,Lt,Dt,Ls,Ds
                
        elif layout_type == "attention":
            # Optimize attention patterns
            if x.ndim == 4:  # BSNH format
                # Similar tiling strategy for attention
                B, S, N, H = x.shape
                tile_size = 128
                
                # Pad attention dimensions
                pad_S = (tile_size - S % tile_size) % tile_size
                pad_H = (tile_size - H % tile_size) % tile_size
                
                if pad_S > 0 or pad_H > 0:
                    x = jnp.pad(x, ((0,0), (0,pad_S), (0,0), (0,pad_H)))
                
                # Reorganize for efficient attention computation
                S_tiles = (S + pad_S) // tile_size
                H_tiles = (H + pad_H) // tile_size
                x = x.reshape(B, S_tiles, tile_size, N, H_tiles, tile_size)
                return x.transpose(0, 1, 3, 4, 2, 5)  # B,St,N,Ht,Ss,Hs
        
        elif layout_type == "mlp":
            # Optimize for MLP operations
            if x.ndim == 3:  # BLH format
                B, L, H = x.shape
                tile_size = 128
                
                # TPU-friendly padding
                pad_L = (tile_size - L % tile_size) % tile_size
                pad_H = (tile_size - H % tile_size) % tile_size
                
                if pad_L > 0 or pad_H > 0:
                    x = jnp.pad(x, ((0,0), (0,pad_L), (0,pad_H)))
                
                # Reshape for efficient MLP computation
                L_tiles = (L + pad_L) // tile_size
                H_tiles = (H + pad_H) // tile_size
                x = x.reshape(B, L_tiles, tile_size, H_tiles, tile_size)
                return x.transpose(0, 1, 3, 2, 4)  # B,Lt,Ht,Ls,Hs
        
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

    def optimize_communication(
        self,
        ops: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Optimize communication patterns across TPU mesh."""
        optimized_ops = []
        current_batch = []
        
        for op in ops:
            if len(current_batch) >= 4:  # Max fusion batch size
                # Fuse compatible operations
                fused_op = self._fuse_communication_ops(current_batch)
                optimized_ops.append(fused_op)
                current_batch = []
            
            if self._can_batch_with(current_batch, op):
                current_batch.append(op)
            else:
                if current_batch:
                    fused_op = self._fuse_communication_ops(current_batch)
                    optimized_ops.append(fused_op)
                current_batch = [op]
        
        if current_batch:
            fused_op = self._fuse_communication_ops(current_batch)
            optimized_ops.append(fused_op)
            
        return optimized_ops

    def _can_batch_with(
        self,
        batch: List[Dict[str, Any]],
        op: Dict[str, Any]
    ) -> bool:
        """Check if operation can be batched with current group."""
        if not batch:
            return True
            
        # Check compatibility criteria
        base_op = batch[0]
        return (
            op["type"] == base_op["type"] and
            op.get("axis") == base_op.get("axis") and
            op.get("reduce_type") == base_op.get("reduce_type")
        )
    
    def _fuse_communication_ops(
        self,
        ops: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Fuse compatible communication operations."""
        if not ops:
            return None
            
        base_op = ops[0]
        if base_op["type"] == "all_reduce":
            # Concatenate tensors for fused all-reduce
            tensors = [op["tensor"] for op in ops]
            fused_tensor = jnp.concatenate(tensors, axis=base_op.get("axis", 0))
            
            return {
                "type": "all_reduce",
                "tensor": fused_tensor,
                "reduce_type": base_op["reduce_type"],
                "num_fused": len(ops)
            }
        
        return base_op  # Return as-is if fusion not possible
    
    def update_performance_metrics(
        self,
        step_metrics: Dict[str, float]
    ) -> None:
        """Update TPU performance monitoring metrics."""
        # Track HBM memory usage
        self.perf_metrics["hbm_usage"].append(step_metrics.get("hbm_usage", 0.0))
        
        # Track compute utilization
        self.perf_metrics["compute_util"].append(step_metrics.get("compute_util", 0.0))
        
        # Maintain sliding window
        window_size = 100
        if len(self.perf_metrics["hbm_usage"]) > window_size:
            self.perf_metrics["hbm_usage"] = self.perf_metrics["hbm_usage"][-window_size:]
            self.perf_metrics["compute_util"] = self.perf_metrics["compute_util"][-window_size:]
    
    def get_performance_stats(self) -> Dict[str, float]:
        """Get current TPU performance statistics."""
        if not self.perf_metrics["hbm_usage"]:
            return {}
            
        return {
            "avg_hbm_usage": sum(self.perf_metrics["hbm_usage"]) / len(self.perf_metrics["hbm_usage"]),
            "max_hbm_usage": max(self.perf_metrics["hbm_usage"]),
            "avg_compute_util": sum(self.perf_metrics["compute_util"]) / len(self.perf_metrics["compute_util"]),
            "min_compute_util": min(self.perf_metrics["compute_util"])
        }
    
    def update_credit_usage(self):
        """Update credit usage metrics."""
        current_time = time.time()
        hours_elapsed = (current_time - self.credit_metrics["last_update"]) / 3600
        
        # Update compute hours (8 cores)
        self.credit_metrics["compute_hours"] += 8 * hours_elapsed
        
        # Update memory GB hours (8GB per core)
        self.credit_metrics["memory_gb_hours"] += 64 * hours_elapsed
        
        self.credit_metrics["last_update"] = current_time
    
    def get_credit_metrics(self) -> Dict[str, float]:
        """Get current credit usage metrics."""
        self.update_credit_usage()
        return {
            "compute_hours": self.credit_metrics["compute_hours"],
            "memory_gb_hours": self.credit_metrics["memory_gb_hours"],
            "estimated_cost": self.credit_metrics["compute_hours"] * 0.35  # Approximate TPU v3 cost per core hour
        }
    
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