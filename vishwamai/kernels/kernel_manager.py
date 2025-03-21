"""Structured kernel management system for VishwamAI."""

from enum import Enum
from typing import Dict, Any, Optional, Type, Union
import jax
import jax.numpy as jnp
import torch
import dataclasses
from abc import ABC, abstractmethod

class KernelType(Enum):
    """Available kernel types."""
    MATMUL = "matmul"
    ATTENTION = "attention" 
    SPARSE = "sparse"
    TREE = "tree"
    FLASH = "flash"
    HYBRID = "hybrid"

class HardwareType(Enum):
    """Supported hardware platforms."""
    TPU = "tpu"
    GPU = "gpu"
    CPU = "cpu"

@dataclasses.dataclass
class KernelConfig:
    """Configuration for kernel execution."""
    hardware: HardwareType
    precision: str = "fp32"
    block_size: int = 128
    use_fp8: bool = False
    use_flash_attn: bool = False
    dynamic_scale: bool = True
    num_warps: int = 8
    profile: bool = False

class Kernel(ABC):
    """Base class for all kernels."""
    
    def __init__(self, config: KernelConfig):
        self.config = config
        self._validate_config()
        
    @abstractmethod
    def forward(self, *args, **kwargs):
        """Forward pass implementation."""
        pass
        
    @abstractmethod
    def backward(self, *args, **kwargs):
        """Backward pass implementation."""
        pass
        
    def _validate_config(self):
        """Validate kernel configuration."""
        if self.config.hardware == HardwareType.TPU:
            # TPU specific validation
            if self.config.block_size % 128 != 0:
                raise ValueError("TPU block size must be multiple of 128")
        elif self.config.hardware == HardwareType.GPU:
            # GPU specific validation
            if self.config.block_size % 32 != 0:
                raise ValueError("GPU block size must be multiple of 32")

class KernelManager:
    """Manages kernel registration and dispatch."""
    
    def __init__(self):
        self._kernels: Dict[KernelType, Dict[HardwareType, Type[Kernel]]] = {}
        self._configs: Dict[HardwareType, KernelConfig] = {}
        self._initialize_configs()
        
    def _initialize_configs(self):
        """Initialize default configurations for each hardware type."""
        self._configs[HardwareType.TPU] = KernelConfig(
            hardware=HardwareType.TPU,
            precision="bf16",
            block_size=128,
            use_fp8=True
        )
        
        self._configs[HardwareType.GPU] = KernelConfig(
            hardware=HardwareType.GPU,
            precision="fp16",
            block_size=64,
            use_flash_attn=True
        )
        
        self._configs[HardwareType.CPU] = KernelConfig(
            hardware=HardwareType.CPU,
            precision="fp32",
            block_size=32
        )
        
    def register_kernel(self, 
                       kernel_cls: Type[Kernel],
                       kernel_type: KernelType,
                       hardware: HardwareType):
        """Register a kernel implementation."""
        if kernel_type not in self._kernels:
            self._kernels[kernel_type] = {}
        self._kernels[kernel_type][hardware] = kernel_cls
        
    def get_kernel(self,
                  kernel_type: KernelType,
                  hardware: Optional[HardwareType] = None,
                  config: Optional[KernelConfig] = None) -> Kernel:
        """Get appropriate kernel implementation."""
        if hardware is None:
            hardware = self._detect_hardware()
            
        if kernel_type not in self._kernels:
            raise KeyError(f"No kernels registered for type: {kernel_type}")
            
        if hardware not in self._kernels[kernel_type]:
            raise KeyError(f"No {kernel_type} kernel for hardware: {hardware}")
            
        if config is None:
            config = self._configs[hardware]
            
        return self._kernels[kernel_type][hardware](config)
    
    def _detect_hardware(self) -> HardwareType:
        """Detect available hardware."""
        try:
            if jax.devices()[0].platform == "tpu":
                return HardwareType.TPU
        except:
            pass
            
        if torch.cuda.is_available():
            return HardwareType.GPU
            
        return HardwareType.CPU
        
    def update_config(self, hardware: HardwareType, **kwargs):
        """Update config for specific hardware."""
        config = self._configs[hardware]
        for key, value in kwargs.items():
            if hasattr(config, key):
                setattr(config, key, value)
                
    def profile_kernel(self,
                      kernel_type: KernelType,
                      hardware: HardwareType,
                      *args,
                      **kwargs) -> Dict[str, float]:
        """Profile kernel performance."""
        config = self._configs[hardware]
        config.profile = True
        kernel = self.get_kernel(kernel_type, hardware, config)
        
        # Run warmup passes
        for _ in range(10):
            kernel.forward(*args, **kwargs)
            
        # Run profiling passes
        times = []
        for _ in range(100):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            
            start.record()
            kernel.forward(*args, **kwargs)
            end.record()
            
            torch.cuda.synchronize()
            times.append(start.elapsed_time(end))
            
        return {
            "mean_ms": sum(times) / len(times),
            "min_ms": min(times),
            "max_ms": max(times)
        }

# Global kernel manager instance
_kernel_manager = KernelManager()

def get_kernel_manager() -> KernelManager:
    """Get the global kernel manager instance."""
    return _kernel_manager