# /home/kasinadhsarma/VishwamAI/vishwamai/inference/optimized_inference.py
"""
Optimized inference interface for VishwamAI models.
Handles device and precision settings for efficient inference on TPUs and GPUs.
"""

import torch
import logging
from typing import Union, Optional, Dict, Any
from vishwamai.optimisation.memory_optimization import MemoryOptimizer
from vishwamai.optimisation.performance_tuning import PerformanceTuner

# Setup TPU support flags
HAS_JAX = False
HAS_TORCH_XLA = False
HAS_TPU = False

# Try to import JAX for TPU support
try:
    import jax
    import jax.numpy as jnp
    from jax import random, jit
    import flax.linen as nn
    HAS_JAX = True
    HAS_TPU = True
except ImportError:
    jax = None
    jnp = None
    nn = None

# Try to import PyTorch XLA as fallback for TPU support
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    HAS_TORCH_XLA = True
    if not HAS_TPU:  # Only set if JAX import failed
        HAS_TPU = True
except ImportError:
    xm = None

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OptimizedInference:
    """
    Interface for optimizing model inference across different devices and precisions.
    Supports both TPU (via JAX/XLA) and GPU optimizations.
    """
    def __init__(self):
        self.device = None
        self.precision = None
        self.device_type = None
        self.memory_optimizer = None
        self.performance_tuner = None
        self._jax_model = None
        self._torch_model = None
        logger.info("Initialized OptimizedInference")

    def set_device(self, device_type: str):
        """Set device with graceful fallback."""
        self.device_type = device_type.lower()
        
        if self.device_type == 'tpu':
            if not HAS_TPU:
                logger.warning("No TPU support found. Falling back to GPU/CPU.")
                self.device_type = 'gpu' if torch.cuda.is_available() else 'cpu'
            else:
                try:
                    if HAS_JAX:
                        self.device = jax.devices("tpu")[0]
                    elif HAS_TORCH_XLA:
                        self.device = xm.xla_device()
                    else:
                        raise RuntimeError("No TPU device found")
                except Exception as e:
                    logger.warning(f"TPU initialization failed: {e}. Falling back to GPU/CPU.")
                    self.device_type = 'gpu' if torch.cuda.is_available() else 'cpu'
        
        # Handle GPU/CPU cases
        if self.device_type == 'gpu':
            if not torch.cuda.is_available():
                logger.warning("CUDA is not available. Falling back to CPU.")
                self.device_type = 'cpu'
                self.device = torch.device('cpu')
            else:
                self.device = torch.device('cuda')
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
        elif self.device_type == 'cpu':
            self.device = torch.device('cpu')
            
        logger.info(f"Device set to: {self.device} ({self.device_type})")

    def set_precision(self, precision: str):
        """
        Set the precision for computations with device-specific optimizations.

        Args:
            precision (str): 'fp16', 'bf16', or 'fp32'.
        """
        supported_precisions = {
            'tpu': ['bf16', 'fp32'],
            'gpu': ['fp16', 'bf16', 'fp32'],
            'cpu': ['fp32']
        }
        
        if precision not in supported_precisions.get(self.device_type, []):
            raise ValueError(f"Unsupported precision {precision} for device {self.device_type}")
            
        self.precision = precision
        logger.info(f"Precision set to: {precision}")

    def optimize_model(self, model):
        """
        Optimize the model for the set device and precision with advanced optimizations.

        Args:
            model: PyTorch or JAX model to optimize.
        """
        if self.device is None or self.precision is None:
            raise ValueError("Device and precision must be set before optimization")

        # Initialize optimizers if not already done
        if not self.memory_optimizer:
            self.memory_optimizer = MemoryOptimizer(model, self.device_type)
        if not self.performance_tuner:
            self.performance_tuner = PerformanceTuner(model, self.device_type)

        if self.device_type == 'tpu':
            if HAS_JAX:
                # Convert to JAX if needed
                if hasattr(model, '__class__') and not any(
                    isinstance(model, t) for t in (jax.Array, nn.Module)
                ):
                    # Convert PyTorch model to JAX (simplified example)
                    self._torch_model = model
                    self._jax_model = self._convert_to_jax(model)
                    model = self._jax_model
                
                # Apply TPU-specific optimizations
                model = self.performance_tuner.optimize_inference(model, use_jit=True)
            elif HAS_TORCH_XLA:
                model.to(xm.xla_device())
            else:
                raise RuntimeError("No TPU support available. Install JAX or PyTorch XLA.")
        else:
            model.to(self.device)
            
            # Apply precision settings
            if self.precision == 'fp16':
                model.half()
            elif self.precision == 'bf16' and self.device_type == 'gpu':
                if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
                    model.bfloat16()
                else:
                    logger.warning("BF16 not supported on this GPU, falling back to FP32")

        # Apply memory optimizations
        self.memory_optimizer.enable_mixed_precision()
        if hasattr(model, 'train'):
            model.eval()
            
        logger.info(f"Model optimized for {self.device_type} inference with {self.precision} precision")
        return model

    def run_model(self, model, input_data: Union[torch.Tensor, "jnp.ndarray"], 
                 temperature: float = 1.0, **kwargs) -> Union[torch.Tensor, "jnp.ndarray"]:
        """
        Run the optimized model with enhanced error handling and device-specific optimizations.

        Args:
            model: The optimized model (PyTorch or JAX).
            input_data: Input tensor or array.
            temperature (float): Temperature for softmax (default: 1.0).
            **kwargs: Additional arguments for model inference.

        Returns:
            Union[torch.Tensor, "jnp.ndarray"]: Model output.
        """
        if self.device_type == 'tpu':
            if HAS_JAX:
                # JAX TPU execution
                @jit
                def _run_with_temp(x, temp):
                    logits = model(x, **kwargs)
                    if temp != 1.0:
                        logits = logits / temp
                        logits = jax.nn.softmax(logits, axis=-1)
                    return logits
                
                # Convert input to JAX array if needed
                if isinstance(input_data, torch.Tensor):
                    input_data = jnp.array(input_data.cpu().numpy())
                
                with jax.default_device(self.device):
                    output = _run_with_temp(input_data, temperature)
            else:
                # PyTorch XLA execution
                with torch.no_grad():
                    output = model(input_data)
                    if temperature != 1.0:
                        output = torch.softmax(output / temperature, dim=-1)
                xm.mark_step()
        else:
            # GPU/CPU execution with PyTorch
            with torch.no_grad(), torch.cuda.amp.autocast(enabled=self.precision in ['fp16', 'bf16']):
                output = model(input_data)
                if temperature != 1.0:
                    output = torch.softmax(output / temperature, dim=-1)
                
        return output

    def _convert_to_jax(self, torch_model):
        """Convert PyTorch model to JAX/Flax (placeholder for actual implementation)."""
        # This would be a complex implementation to convert PyTorch models to JAX
        # For now, we just raise an error
        raise NotImplementedError("PyTorch to JAX model conversion not yet implemented")

    def get_memory_stats(self) -> Dict[str, Any]:
        """Get current memory usage statistics for the device."""
        return self.memory_optimizer.get_memory_usage() if self.memory_optimizer else {}

    def clear_cache(self):
        """Clear device memory cache."""
        if self.memory_optimizer:
            self.memory_optimizer.clear_device_memory()

if __name__ == "__main__":
    # Example usage with memory monitoring
    class DummyModel(torch.nn.Module):
        def forward(self, x):
            return torch.tensor([ord('t'), ord('e'), ord('s'), ord('t')], device=x.device)

    model = DummyModel()
    opt_inf = OptimizedInference()
    
    # Try TPU first, fall back to GPU/CPU
    try:
        opt_inf.set_device('tpu')
        opt_inf.set_precision('bf16')
    except (ImportError, RuntimeError) as e:
        logger.warning(f"TPU initialization failed: {e}. Falling back to GPU/CPU")
        opt_inf.set_device('gpu' if torch.cuda.is_available() else 'cpu')
        opt_inf.set_precision('fp16' if torch.cuda.is_available() else 'fp32')

    model = opt_inf.optimize_model(model)
    
    # Monitor memory usage
    print("Initial memory stats:", opt_inf.get_memory_stats())
    
    input_tensor = torch.tensor([1, 2, 3], device=opt_inf.device).unsqueeze(0)
    output = opt_inf.run_model(model, input_tensor)
    
    print("After inference memory stats:", opt_inf.get_memory_stats())
    print("Output:", ''.join(chr(int(x)) for x in output))