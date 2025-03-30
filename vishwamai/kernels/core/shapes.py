"""Dynamic shape handling and optimization system."""

from typing import Dict, Any, Optional, Tuple, List, Union
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import torch
import numpy as np

from .kernel import HardwareType, KernelConfig

@dataclass
class ShapeSpec:
    """Specification for tensor shapes."""
    dims: Tuple[Union[int, str], ...]
    min_sizes: Optional[Tuple[int, ...]] = None
    max_sizes: Optional[Tuple[int, ...]] = None
    dynamic_axes: Tuple[int, ...] = ()
    
class ShapeTracker:
    """Track and optimize tensor shapes."""
    
    def __init__(self, config: KernelConfig):
        self.config = config
        self.shape_cache: Dict[str, ShapeSpec] = {}
        
    def register_shape(self,
                      name: str,
                      spec: ShapeSpec):
        """Register shape specification."""
        self.shape_cache[name] = spec
        
    def optimize_shape(self,
                      shape: Tuple[int, ...],
                      spec: Optional[ShapeSpec] = None) -> Tuple[int, ...]:
        """Optimize shape for target hardware."""
        if self.config.hardware == HardwareType.TPU:
            return self._optimize_tpu_shape(shape, spec)
        elif self.config.hardware == HardwareType.GPU:
            return self._optimize_gpu_shape(shape, spec)
        return shape
        
    def _optimize_tpu_shape(self,
                           shape: Tuple[int, ...],
                           spec: Optional[ShapeSpec]) -> Tuple[int, ...]:
        """Optimize shapes for TPU execution."""
        # Pad to TPU-efficient dimensions
        optimized = []
        for i, dim in enumerate(shape):
            if spec and i in spec.dynamic_axes:
                # Handle dynamic dimension
                padded = self._next_tpu_efficient_size(dim)
                optimized.append(padded)
            else:
                # Static dimension
                if dim % 128 != 0:
                    padded = ((dim + 127) // 128) * 128
                    optimized.append(padded)
                else:
                    optimized.append(dim)
                    
        return tuple(optimized)
        
    def _optimize_gpu_shape(self,
                           shape: Tuple[int, ...],
                           spec: Optional[ShapeSpec]) -> Tuple[int, ...]:
        """Optimize shapes for GPU execution."""
        # Pad to warp-efficient dimensions
        optimized = []
        for i, dim in enumerate(shape):
            if spec and i in spec.dynamic_axes:
                # Handle dynamic dimension
                padded = self._next_gpu_efficient_size(dim)
                optimized.append(padded)
            else:
                # Static dimension
                if dim % 32 != 0:
                    padded = ((dim + 31) // 32) * 32
                    optimized.append(padded)
                else:
                    optimized.append(dim)
                    
        return tuple(optimized)
        
    def _next_tpu_efficient_size(self, size: int) -> int:
        """Get next TPU-efficient size."""
        return ((size + 127) // 128) * 128
        
    def _next_gpu_efficient_size(self, size: int) -> int:
        """Get next GPU-efficient size."""
        return ((size + 31) // 32) * 32
        
    def create_dynamic_shape_handler(self,
                                   static_shapes: Dict[str, Tuple[int, ...]],
                                   dynamic_axes: Dict[str, Tuple[int, ...]]):
        """Create handler for dynamic shapes."""
        if self.config.hardware == HardwareType.TPU:
            return self._create_tpu_shape_handler(static_shapes, dynamic_axes)
        elif self.config.hardware == HardwareType.GPU:
            return self._create_gpu_shape_handler(static_shapes, dynamic_axes)
            
    def _create_tpu_shape_handler(self,
                                 static_shapes: Dict[str, Tuple[int, ...]],
                                 dynamic_axes: Dict[str, Tuple[int, ...]]):
        """Create TPU-specific dynamic shape handler."""
        
        def get_shape_update(name: str,
                            runtime_shape: Tuple[int, ...]) -> Dict[str, Any]:
            """Get shape updates for TPU compilation."""
            static_shape = static_shapes[name]
            dynamic_dims = dynamic_axes.get(name, ())
            
            # Create dynamic dimension bindings
            dynamic_sizes = {
                f"d{i}": size
                for i, size in enumerate(runtime_shape)
                if i in dynamic_dims
            }
            
            # Create JAX shape polymorphic constraints
            constraints = {}
            for i, size in enumerate(runtime_shape):
                if i in dynamic_dims:
                    dim_name = f"d{i}"
                    # Add size constraints
                    constraints[dim_name] = (
                        1,  # Minimum size
                        None  # No maximum (TPU can handle any size)
                    )
                    
            return {
                "dynamic_sizes": dynamic_sizes,
                "constraints": constraints
            }
            
        return get_shape_update
        
    def _create_gpu_shape_handler(self,
                                 static_shapes: Dict[str, Tuple[int, ...]],
                                 dynamic_axes: Dict[str, Tuple[int, ...]]):
        """Create GPU-specific dynamic shape handler."""
        
        def get_shape_update(name: str,
                            runtime_shape: Tuple[int, ...]) -> Dict[str, Any]:
            """Get shape updates for TorchScript compilation."""
            static_shape = static_shapes[name]
            dynamic_dims = dynamic_axes.get(name, ())
            
            # Create TorchScript dynamic shapes
            script_shapes = []
            for i, (static, runtime) in enumerate(zip(static_shape, runtime_shape)):
                if i in dynamic_dims:
                    script_shapes.append(-1)  # Dynamic dimension
                else:
                    script_shapes.append(static)
                    
            return {
                "script_shape": script_shapes,
                "dynamic_dims": dynamic_dims
            }
            
        return get_shape_update
        
class DynamicShapeOptimizer:
    """Optimize kernels for dynamic shapes."""
    
    def __init__(self, config: KernelConfig):
        self.config = config
        self.shape_tracker = ShapeTracker(config)
        
    def prepare_dynamic_kernel(self,
                             kernel_fn: Any,
                             input_specs: Dict[str, ShapeSpec]) -> Any:
        """Prepare kernel for dynamic shape handling."""
        if self.config.hardware == HardwareType.TPU:
            return self._prepare_tpu_kernel(kernel_fn, input_specs)
        elif self.config.hardware == HardwareType.GPU:
            return self._prepare_gpu_kernel(kernel_fn, input_specs)
        return kernel_fn
        
    def _prepare_tpu_kernel(self,
                           kernel_fn: Any,
                           input_specs: Dict[str, ShapeSpec]) -> Any:
        """Prepare TPU kernel for dynamic shapes."""
        
        def wrapped_kernel(**inputs):
            # Get runtime shapes
            runtime_shapes = {
                name: tensor.shape
                for name, tensor in inputs.items()
            }
            
            # Create shape handler
            handler = self.shape_tracker.create_dynamic_shape_handler(
                {name: spec.dims for name, spec in input_specs.items()},
                {name: spec.dynamic_axes for name, spec in input_specs.items()}
            )
            
            # Update shapes
            shape_updates = {
                name: handler(name, shape)
                for name, shape in runtime_shapes.items()
            }
            
            # Create padded inputs
            padded_inputs = {}
            for name, tensor in inputs.items():
                if name in shape_updates:
                    # Pad to efficient size
                    update = shape_updates[name]
                    padded_shape = self.shape_tracker.optimize_shape(
                        tensor.shape,
                        input_specs[name]
                    )
                    
                    # Create padded tensor
                    if len(padded_shape) > len(tensor.shape):
                        # Need padding
                        padding = [(0, p - s) for s, p in zip(tensor.shape, padded_shape)]
                        padded_inputs[name] = jnp.pad(tensor, padding)
                    else:
                        padded_inputs[name] = tensor
                else:
                    padded_inputs[name] = tensor
                    
            # Run kernel with padded inputs
            result = kernel_fn(**padded_inputs)
            
            # Remove padding from result
            if isinstance(result, tuple):
                return tuple(
                    output[:original.shape[0]]
                    if len(output.shape) == len(original.shape)
                    else output
                    for output, original in zip(result, inputs.values())
                )
            else:
                first_input = next(iter(inputs.values()))
                return result[:first_input.shape[0]]
                
        return wrapped_kernel
        
    def _prepare_gpu_kernel(self,
                           kernel_fn: Any,
                           input_specs: Dict[str, ShapeSpec]) -> Any:
        """Prepare GPU kernel for dynamic shapes."""
        
        def wrapped_kernel(**inputs):
            # Get runtime shapes
            runtime_shapes = {
                name: tuple(tensor.size())
                for name, tensor in inputs.items()
            }
            
            # Create shape handler
            handler = self.shape_tracker.create_dynamic_shape_handler(
                {name: spec.dims for name, spec in input_specs.items()},
                {name: spec.dynamic_axes for name, spec in input_specs.items()}
            )
            
            # Update shapes
            shape_updates = {
                name: handler(name, shape)
                for name, shape in runtime_shapes.items()
            }
            
            # Create padded inputs
            padded_inputs = {}
            for name, tensor in inputs.items():
                if name in shape_updates:
                    # Pad to efficient size
                    update = shape_updates[name]
                    padded_shape = self.shape_tracker.optimize_shape(
                        tuple(tensor.size()),
                        input_specs[name]
                    )
                    
                    # Create padded tensor
                    if any(p > s for p, s in zip(padded_shape, tensor.size())):
                        # Need padding
                        padding = []
                        for s, p in zip(reversed(tensor.size()), reversed(padded_shape)):
                            padding.extend([0, p - s])
                        padded_inputs[name] = torch.nn.functional.pad(tensor, padding)
                    else:
                        padded_inputs[name] = tensor
                else:
                    padded_inputs[name] = tensor
                    
            # Run kernel with padded inputs
            with torch.cuda.amp.autocast():
                result = kernel_fn(**padded_inputs)
                
            # Remove padding from result
            if isinstance(result, tuple):
                return tuple(
                    output[:original.size(0)]
                    if output.dim() == original.dim()
                    else output
                    for output, original in zip(result, inputs.values())
                )
            else:
                first_input = next(iter(inputs.values()))
                return result[:first_input.size(0)]
                
        return wrapped_kernel