"""Decorators for JIT kernel compilation."""

import functools
import inspect
from typing import Callable, Optional, Dict, Any, TypeVar, Union
from dataclasses import dataclass

from jax import jit, device_put
from jax.lib import xla_client

from .compiler import KernelPlatform, KernelTemplate, compile_kernel
from .templates import get_default_template
from vishwamai.kernels.core.kernel import KernelConfig

F = TypeVar('F', bound=Callable)

@dataclass
class KernelDecorator:
    platform: str
    config: Optional[KernelConfig] = None
    template: Optional[Union[str, KernelTemplate]] = None
    compile_options: Optional[Dict[str, Any]] = None

def kernel(name: Optional[str] = None, platform: str = KernelPlatform.TPU,
          config: Optional[KernelConfig] = None,
          template: Optional[Union[str, KernelTemplate]] = None,
          compile_options: Optional[Dict[str, Any]] = None,
          auto_optimize: bool = True) -> Callable[[F], F]:
    def decorator(fn: F) -> F:
        kernel_name = name or fn.__name__
        sig = inspect.signature(fn)
        arg_types = tuple(p.annotation for p in sig.parameters.values() if p.annotation != inspect.Parameter.empty)
        decorator_config = KernelDecorator(platform=platform, config=config, template=template, compile_options=compile_options)
        
        @functools.wraps(fn)
        def wrapped(*args, **kwargs):
            if hasattr(fn, '_no_jit'): return fn(*args, **kwargs)
            template_args = {'fn_source': inspect.getsource(fn)}
            if decorator_config.compile_options: template_args.update(decorator_config.compile_options)
            
            kernel_template = decorator_config.template or get_default_template(kernel_name, platform, arg_types, auto_optimize)
            
            if platform == KernelPlatform.TPU and auto_optimize:
                args = tuple(device_put(arg, "tpu") for arg in args)
                if decorator_config.config is None:
                    decorator_config.config = KernelConfig(block_size=128, use_bfloat16=True, precision=xla_client.PrecisionConfig.HIGH)
            
            kernel_fn = compile_kernel(kernel_name, arg_types, kernel_template, platform=platform,
                                    config=decorator_config.config, template_args=template_args)
            if platform == KernelPlatform.TPU: kernel_fn = jit(kernel_fn)
            return kernel_fn(*args, **kwargs)
        
        wrapped._kernel_config = decorator_config
        return wrapped
    return decorator

def no_jit(fn: F) -> F:
    fn._no_jit = True
    return fn

def tpu_kernel(name: Optional[str] = None, config: Optional[KernelConfig] = None, auto_optimize: bool = True) -> Callable[[F], F]:
    return kernel(name, KernelPlatform.TPU, config or KernelConfig(block_size=128, use_bfloat16=True, precision=xla_client.PrecisionConfig.HIGH), auto_optimize=auto_optimize)

def gpu_kernel(name: Optional[str] = None, config: Optional[KernelConfig] = None, use_tensor_cores: bool = True) -> Callable[[F], F]:
    return kernel(name, KernelPlatform.GPU, config or KernelConfig(block_size=64, use_fp16=True), compile_options={'use_tensor_cores': use_tensor_cores})

def sparse_kernel(name: Optional[str] = None, platform: str = KernelPlatform.TPU, block_size: int = 128, min_sparsity: float = 0.8) -> Callable[[F], F]:
    return kernel(name, platform, KernelConfig(block_size=block_size, use_bfloat16=platform == KernelPlatform.TPU), compile_options={'min_sparsity': min_sparsity})

def expert_kernel(name: Optional[str] = None, platform: str = KernelPlatform.TPU, num_experts: int = 8) -> Callable[[F], F]:
    return kernel(name, platform, KernelConfig(block_size=128 if platform == KernelPlatform.TPU else 64, use_bfloat16=platform == KernelPlatform.TPU), compile_options={'num_experts': num_experts})

def parallel_kernel(name: Optional[str] = None, platform: str = KernelPlatform.TPU, parallel_dim: int = 0, chunk_size: Optional[int] = None) -> Callable[[F], F]:
    return kernel(name, platform, KernelConfig(block_size=128 if platform == KernelPlatform.TPU else 64, use_bfloat16=platform == KernelPlatform.TPU), compile_options={'parallel_dim': parallel_dim, 'chunk_size': chunk_size})

def attention_kernel(name: Optional[str] = None, platform: str = KernelPlatform.TPU, use_flash: bool = True, causal: bool = False, window_size: Optional[int] = None) -> Callable[[F], F]:
    return kernel(name, platform, KernelConfig(block_size=128 if platform == KernelPlatform.TPU else 64, use_bfloat16=platform == KernelPlatform.TPU, use_efficient_attention=True), compile_options={'use_flash': use_flash, 'causal': causal, 'window_size': window_size})
