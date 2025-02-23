"""TPU utilities and optimizations for JAX training."""
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import os
from functools import partial

import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec as P
import numpy as np

def create_device_mesh(
    mesh_shape: Optional[Tuple[int, ...]] = None,
    axis_names: Optional[List[str]] = None
) -> Mesh:
    """Create TPU device mesh for optimal sharding.
    
    Args:
        mesh_shape: Optional shape for device mesh
        axis_names: Optional names for mesh axes
        
    Returns:
        JAX device mesh
    """
    devices = jax.devices()
    
    if mesh_shape is None:
        # Automatically determine mesh shape
        n = int(len(devices) ** 0.5)
        mesh_shape = (n, n)
        
    if axis_names is None:
        axis_names = ['data', 'model']
        
    device_mesh = mesh_utils.create_device_mesh(mesh_shape)
    return Mesh(device_mesh, axis_names)

def init_data_parallelism():
    """Initialize pure data parallel training."""
    devices = jax.devices()
    return Mesh(
        np.array(devices),
        ['data']
    )

def init_model_parallelism(
    model_parallel_submesh: Optional[Tuple[int, ...]] = None
):
    """Initialize model parallel training.
    
    Args:
        model_parallel_submesh: Optional shape for model parallel submesh
    """
    devices = jax.devices()
    if model_parallel_submesh is None:
        n = int(len(devices) ** 0.5)
        model_parallel_submesh = (1, n, n)
        
    return Mesh(
        mesh_utils.create_device_mesh(model_parallel_submesh),
        ['replica', 'data', 'model']
    )

def get_sharding_config(
    batch_size: int,
    seq_length: int,
    hidden_dim: int,
    num_heads: int,
    vocab_size: int,
    mesh_shape: Optional[Tuple[int, ...]] = None
) -> Dict[str, P]:
    """Get sharding configuration for model parameters.
    
    Args:
        batch_size: Batch size per device
        seq_length: Maximum sequence length
        hidden_dim: Hidden dimension size
        num_heads: Number of attention heads
        vocab_size: Vocabulary size
        mesh_shape: Optional mesh shape
        
    Returns:
        Dictionary of parameter sharding specs
    """
    if mesh_shape is None:
        n = int(jax.device_count() ** 0.5)
        mesh_shape = (n, n)
        
    # Default 2D sharding strategy
    return {
        'embedding': P('model', 'data'),
        'attention': {
            'query': P('model', 'data'),
            'key': P('model', 'data'),
            'value': P('model', 'data'),
            'output': P('data', 'model')
        },
        'mlp': {
            'wi': P('model', 'data'),
            'wo': P('data', 'model')
        },
        'layer_norm': P(None),
        'output': P('model', 'data')
    }

def get_jit_config() -> Dict[str, Any]:
    """Get JIT compilation configuration for TPU."""
    return {
        'jit_device_inputs': True,
        'xla_cpu_fast_math_honor_infs': True,
        'xla_cpu_fast_math_honor_nans': True,
        'xla_force_host_platform_device_count': jax.device_count()
    }

@partial(jax.jit, static_argnums=(1, 2))
def split_batch_to_devices(
    batch: Dict[str, jnp.ndarray],
    batch_size: int,
    num_devices: Optional[int] = None
) -> Dict[str, jnp.ndarray]:
    """Split batch across devices.
    
    Args:
        batch: Input batch dictionary
        batch_size: Global batch size
        num_devices: Optional number of devices
        
    Returns:
        Batch split across devices
    """
    if num_devices is None:
        num_devices = jax.device_count()
        
    local_batch_size = batch_size // num_devices
    
    def split_array(x):
        return x.reshape((num_devices, local_batch_size) + x.shape[1:])
        
    return jax.tree_map(split_array, batch)

def create_parameter_sharding_specs(
    params: Any,
    mesh: Mesh
) -> Any:
    """Create sharding specifications for parameters.
    
    Args:
        params: Model parameters
        mesh: Device mesh
        
    Returns:
        Parameter sharding specs
    """
    def infer_spec(param):
        ndim = len(param.shape)
        if ndim == 0:
            return P()
        elif ndim == 1:
            return P('model')
        elif ndim == 2:
            return P('model', 'data')
        else:
            return P('data') + (None,) * (ndim - 1)
            
    return jax.tree_map(infer_spec, params)

def setup_tpu_cluster():
    """Setup TPU cluster environment."""
    # Detect TPU runtime
    if 'TPU_NAME' in os.environ:
        # Cloud TPU configuration
        tpu = os.environ['TPU_NAME']
        if not tpu.startswith('tpu_worker'):
            tpu = 'tpu_worker:{}'.format(tpu)
        os.environ['GRPC_DEFAULT_SSL_ROOTS_FILE_PATH'] = ''
        os.environ['TPU_API_INIT_TIMEOUT'] = '0'
    else:
        # Local TPU configuration
        tpu = ''
        
    # Initialize JAX runtime
    jax.config.update('jax_xla_backend', 'tpu_driver')
    jax.config.update('jax_backend_target', tpu)
    
    # Log TPU configuration
    devices = jax.devices()
    print(f'Number of devices: {len(devices)}')
    print(f'Device type: {devices[0].device_kind}')

def enable_tpu_logging():
    """Enable TPU-specific logging."""
    jax.config.update('jax_log_compiles', True)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0'
    os.environ['TF_CPP_VMODULE'] = (
        'tensor=5,computation_client=5,'
        'xla_bridge=5,xla_client=5'
    )

def configure_tpu_bf16():
    """Configure bfloat16 on TPU."""
    jax.config.update('jax_default_dtype_bits', 16)
    jax.config.update('jax_enable_x64', False)
    os.environ['JAX_ENABLE_BF16_CONVERSION'] = '1'

def profile_tpu_computation(fn: Callable) -> Callable:
    """Decorator to profile TPU computation.
    
    Args:
        fn: Function to profile
        
    Returns:
        Profiled function
    """
    def wrapped(*args, **kwargs):
        # Start profiling
        with jax.profiler.trace('computation') as trace:
            with jax.profiler.TraceAnnotations() as ta:
                ta.add_metadata(
                    type='computation',
                    name=fn.__name__
                )
                result = fn(*args, **kwargs)
                
        # Log profile
        total_time = trace.get_completion_time()
        print(f'Function {fn.__name__} took {total_time:.2f}ms')
        
        return result
    return wrapped
