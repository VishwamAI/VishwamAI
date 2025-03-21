"""TPU-optimized tensor parallelism utilities."""

import jax
import jax.numpy as jnp
from typing import Dict, Any, Tuple, List, Optional
import flax.linen as nn
import numpy as np

def shard_params(params: Dict[str, Any], mesh: jax.sharding.Mesh, 
                axis_rules: Dict[str, str]) -> Dict[str, Any]:
    """
    Shards model parameters according to tensor parallelism rules.
    
    Args:
        params: Model parameters dictionary
        mesh: JAX device mesh
        axis_rules: Rules for sharding different parameter types
    
    Returns:
        Sharded parameters dictionary
    """
    from jax.sharding import PartitionSpec as P
    
    def _get_partition_spec(name: str, param_shape: Tuple[int, ...]) -> P:
        # Default: no partitioning
        spec = P()
        
        # Apply rules based on parameter name and shape
        for key, rule in axis_rules.items():
            if key in name:
                if rule == "model":
                    # Shard first dimension across model parallel axis
                    spec = P("model", None)
                elif rule == "data":
                    # Shard first dimension across data parallel axis
                    spec = P("data", None)
                elif rule == "model_and_data":
                    # 2D sharding
                    spec = P("model", "data")
                elif rule == "fully_replicated":
                    spec = P()
        
        return spec
    
    # Create sharding specs for all parameters
    partition_specs = {
        name: _get_partition_spec(name, param.shape)
        for name, param in params.items()
    }
    
    # Apply sharding
    sharded_params = {}
    for name, param in params.items():
        spec = partition_specs[name]
        sharded_params[name] = jax.device_put(param, jax.sharding.NamedSharding(mesh, spec))
    
    return sharded_params

def all_gather(x: jnp.ndarray, axis_name: str = "model") -> jnp.ndarray:
    """
    Gather values across all devices along specified axis.
    
    Args:
        x: Input tensor
        axis_name: Name of parallel axis to gather across
    
    Returns:
        Gathered tensor
    """
    return jax.lax.all_gather(x, axis_name=axis_name)

def all_reduce(x: jnp.ndarray, axis_name: str = "model") -> jnp.ndarray:
    """
    Reduce (sum) values across all devices along specified axis.
    
    Args:
        x: Input tensor
        axis_name: Name of parallel axis to reduce across
    
    Returns:
        Reduced tensor
    """
    return jax.lax.psum(x, axis_name=axis_name)

def setup_dp_tp_mesh(num_devices: int, dp_size: Optional[int] = None) -> jax.sharding.Mesh:
    """
    Set up a device mesh for data and tensor parallelism.
    
    Args:
        num_devices: Total number of devices
        dp_size: Number of data parallel replicas (if None, computed automatically)
    
    Returns:
        JAX device mesh
    """
    # Auto-configure if dp_size not provided
    if dp_size is None:
        # Heuristic: square root of devices gives a balanced configuration
        dp_size = int(np.sqrt(num_devices))
        while num_devices % dp_size != 0:
            dp_size -= 1
    
    tp_size = num_devices // dp_size
    
    # Get physical devices
    devices = jax.devices()
    if len(devices) < num_devices:
        raise ValueError(f"Requested {num_devices} devices but only {len(devices)} available")
    
    # Create mesh for data and model (tensor) parallelism
    device_mesh = np.array(devices[:num_devices]).reshape(dp_size, tp_size)
    return jax.sharding.Mesh(device_mesh, axis_names=("data", "model"))

def split_batch_dim(x: jnp.ndarray, batch_dim: int = 0, num_devices: int = 8) -> jnp.ndarray:
    """
    Split batch dimension into virtual batch and device dimensions for TPU efficiency.
    
    Args:
        x: Input tensor
        batch_dim: Dimension to split
        num_devices: Number of devices to shard across
    
    Returns:
        Tensor with split batch dimension
    """
    shape = list(x.shape)
    if shape[batch_dim] % num_devices != 0:
        raise ValueError(f"Batch size {shape[batch_dim]} not divisible by {num_devices} devices")
        
    virtual_batch = shape[batch_dim] // num_devices
    shape.pop(batch_dim)
    shape.insert(batch_dim, num_devices)
    shape.insert(batch_dim + 1, virtual_batch)
    
    return x.reshape(shape)
