"""Core kernel management functionality."""

from .kernel import (
    Kernel,
    KernelType,
    HardwareType,
    KernelConfig,
    AbstractKernel
)

from .kernel_manager import (
    KernelManager,
    get_kernel_manager,
    register_kernel,
    get_kernel
)

__all__ = [
    # Base classes and types
    "Kernel",
    "AbstractKernel",
    "KernelType",
    "HardwareType",
    "KernelConfig",

    # Kernel management
    "KernelManager",
    "get_kernel_manager",
    "register_kernel",
    "get_kernel"
]