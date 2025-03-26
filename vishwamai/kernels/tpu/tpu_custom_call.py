"""TPU-optimized custom call operations for JAX."""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union
import jax
import jax.numpy as jnp
from jax import core
from jax.interpreters import xla
from jax.lib import xla_client
from jax.interpreters import mlir
import numpy as np
from jax._src import abstract_arrays

# Register the TPU custom call primitive
tpu_custom_call_p = core.Primitive("tpu_custom_call")
tpu_custom_call_p.multiple_results = True
tpu_custom_call_p.def_impl(lambda *args, **kwargs: None)

@tpu_custom_call_p.def_abstract_eval
def _tpu_custom_call_abstract_eval(*_, out_avals, **__):
  return out_avals


def _avals_to_layouts(avals) -> Sequence[Sequence[int]]:
  return [tuple(range(a.ndim - 1, -1, -1)) for a in avals]

def tpu_custom_call(
    call_target_name: str,
    out_avals: Sequence[core.ShapedArray],
    operands: Sequence[Any],
    *,
    operand_layouts: Optional[Sequence[Sequence[int]]] = None,
    result_layouts: Optional[Sequence[Sequence[int]]] = None,
    opaque: Optional[bytes] = None,
    has_side_effect: bool = False,
    schedule: Optional[int] = None,
    backend_config: Optional[Dict[str, Any]] = None,
):
    """Makes a TPU custom call using TPU-optimized layouts.
    
    Args:
        call_target_name: Name of the custom TPU kernel
        out_avals: Sequence of output avals
        operands: Sequence of JAX operands
        operand_layouts: Optional layouts for operands
        result_layouts: Optional layouts for results
        opaque: Optional opaque data to pass to the custom call
        has_side_effect: Whether the call has side effects
        schedule: Optional scheduling priority
        backend_config: Optional backend configuration
        
    Returns:
        Result of the TPU custom call
    """
    operand_layouts = operand_layouts or _avals_to_layouts([core.get_aval(x) for x in operands])
    result_layouts = result_layouts or _avals_to_layouts(out_avals)
    backend_config = backend_config or {}
    
    flat_operands, in_tree = jax.tree_util.tree_flatten(operands)
    
    return tpu_custom_call_p.bind(
        *flat_operands,
        call_target_name=call_target_name,
        out_avals=out_avals,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
        opaque=opaque,
        has_side_effect=has_side_effect,
        schedule=schedule,
        backend_config=backend_config,
    )

def tpu_custom_call_lowering(ctx, *operands, call_target_name, out_avals, 
                             operand_layouts, result_layouts, opaque, 
                             has_side_effect, schedule, backend_config):
    """Lowering rule for TPU custom calls."""
    operand_shapes = [mlir.aval_to_ir_type(core.get_aval(op)) for op in operands]
    result_shapes = [mlir.aval_to_ir_type(aval) for aval in out_avals]
    
    flat_operands = [mlir.ir_constants(operand) for operand in operands]
    
    # Convert backend config to bytes
    if backend_config:
        import json
        backend_config_bytes = json.dumps(backend_config).encode('utf-8')
    else:
        backend_config_bytes = None
    
    # Create the TPU optimized custom call
    out = mlir.xla_custom_call(
        call_target_name,
        result_types=result_shapes,
        operands=flat_operands,
        backend_config=backend_config_bytes,
        has_side_effect=has_side_effect,
        operand_layouts=operand_layouts,
        result_layouts=result_layouts,
    )
    
    if len(out_avals) == 1:
        return [out]
    else:
        return list(out)

# Register the lowering rule
mlir.register_lowering(tpu_custom_call_p, tpu_custom_call_lowering)

def compile_tpu_kernel(
    name: str,
    fn: Callable,
    input_shapes: Sequence[Tuple[int, ...]],
    output_shapes: Sequence[Tuple[int, ...]],
    input_dtypes: Sequence[np.dtype],
    output_dtypes: Sequence[np.dtype],
):
    """Compile a TPU kernel with JAX.
    
    Args:
        name: Name of the kernel
        fn: Function to compile
        input_shapes: Shapes of inputs
        output_shapes: Shapes of outputs
        input_dtypes: Data types of inputs
        output_dtypes: Data types of outputs
        
    Returns:
        Compiled TPU kernel
    """
    input_avals = [abstract_arrays.ShapedArray(shape, dtype) 
                   for shape, dtype in zip(input_shapes, input_dtypes)]
    output_avals = [abstract_arrays.ShapedArray(shape, dtype) 
                    for shape, dtype in zip(output_shapes, output_dtypes)]
    
    # Create a JAX-compiled function
    jit_fn = jax.jit(fn)
    
    # Create wrapper that uses custom call
    def custom_call_wrapper(*args):
        # Prepare backend config
        backend_config = {
            "kernel_name": name,
            "input_dtypes": [str(dt) for dt in input_dtypes],
            "output_dtypes": [str(dt) for dt in output_dtypes],
        }
        
        return tpu_custom_call(
            call_target_name=name,
            out_avals=output_avals,
            operands=args,
            backend_config=backend_config
        )
    
    return custom_call_wrapper

# Utility functions for TPU optimization

def optimize_tpu_layout(x: jnp.ndarray) -> jnp.ndarray:
    """Optimize tensor layout for TPU memory access patterns."""
    if x.ndim <= 1:
        return x
    elif x.ndim == 2:
        # For matrix operations, TPUs prefer 128x128 tiling
        return x
    elif x.ndim == 3:
        # For 3D tensors, optimize for TPU's memory hierarchy
        return jnp.asarray(x, order='F')
    elif x.ndim == 4:
        # For 4D tensors (common in attention), use TPU-friendly layout
        return jnp.transpose(x, (0, 2, 1, 3))
    else:
        # For higher dimensions, preserve original layout
        return x

def pad_to_tpu_multiple(x: jnp.ndarray, multiple: int = 128) -> jnp.ndarray:
    """Pad tensor dimensions to be multiples of TPU-preferred sizes."""
    shape = x.shape
    padded_shape = [(s + multiple - 1) // multiple * multiple for s in shape]
    
    if shape == tuple(padded_shape):
        return x
    
    # Create padding configuration
    pad_config = [(0, padded_shape[i] - shape[i]) for i in range(len(shape))]
    return jnp.pad(x, pad_config, mode='constant', constant_values=0)

def get_optimal_tpu_layout(shape: Tuple[int, ...]) -> Sequence[int]:
    """Get the optimal memory layout for a given tensor shape on TPU."""
    ndim = len(shape)
    
    if ndim <= 1:
        return tuple(range(ndim))
    elif ndim == 2:
        # For matrices, minor-to-major order is generally efficient
        return (1, 0)
    elif ndim == 3:
        # For 3D tensors, batch dimension first in major-to-minor
        return (0, 2, 1)
    elif ndim == 4:
        # For 4D tensors used in attention computations
        return (0, 2, 1, 3)
    else:
        # Default to standard minor-to-major order
        return tuple(range(ndim - 1, -1, -1))