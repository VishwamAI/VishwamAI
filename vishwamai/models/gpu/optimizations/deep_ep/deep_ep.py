"""
Python interface for Deep Efficient Parallelism (DeepEP) CUDA operations.
"""

import os
import torch
import ctypes
from ctypes import c_void_p, c_size_t, c_int, c_bool, Structure, POINTER
from typing import Optional, Tuple

# Load the compiled CUDA library
_lib_path = os.path.join(os.path.dirname(__file__), "csrc/libdeep_ep.so")
_lib = None

def _load_library():
    global _lib
    if _lib is None:
        if not os.path.exists(_lib_path):
            raise RuntimeError(
                f"DeepEP CUDA library not found at {_lib_path}. "
                "Please compile the CUDA kernels first."
            )
        _lib = ctypes.CDLL(_lib_path)
        
        # Set function signatures
        _lib.deep_ep_init_buffer.argtypes = [POINTER(DeepEPParams)]
        _lib.deep_ep_init_buffer.restype = ctypes.c_int
        
        _lib.deep_ep_free_buffer.argtypes = [POINTER(DeepEPParams)]
        _lib.deep_ep_free_buffer.restype = ctypes.c_int
        
        _lib.deep_ep_dispatch.argtypes = [
            c_void_p, c_void_p, c_void_p, c_void_p, POINTER(DeepEPParams)
        ]
        _lib.deep_ep_dispatch.restype = ctypes.c_int
        
        _lib.deep_ep_combine.argtypes = [
            c_void_p, c_void_p, c_void_p, c_void_p, POINTER(DeepEPParams)
        ]
        _lib.deep_ep_combine.restype = ctypes.c_int

# Define the parameter structure
class DeepEPParams(Structure):
    _fields_ = [
        ("hidden_buffer", c_void_p),
        ("nvlink_buffer", c_void_p),
        ("rdma_buffer", c_void_p),
        ("hidden_bytes", c_size_t),
        ("nvl_bytes", c_size_t),
        ("rdma_bytes", c_size_t),
        ("num_experts", c_int),
        ("num_tokens", c_int),
        ("hidden_dim", c_int),
        ("async_dispatch", c_bool),
        ("stream", c_void_p),
        ("event", c_void_p)
    ]

class DeepEPModule:
    """Python wrapper for DeepEP CUDA operations"""
    
    def __init__(self, hidden_size: int, num_experts: int):
        _load_library()
        
        self.hidden_size = hidden_size
        self.num_experts = num_experts
        self.params = DeepEPParams()
        self.params.hidden_bytes = hidden_size * 4  # float32
        self.params.num_experts = num_experts
        self.params.async_dispatch = True
        
        # Create CUDA stream and event
        self.stream = torch.cuda.Stream()
        self.event = torch.cuda.Event()
        self.params.stream = self.stream.cuda_stream
        self.params.event = self.event.cuda_event
        
        # Initialize buffers
        err = _lib.deep_ep_init_buffer(ctypes.byref(self.params))
        if err != 0:
            raise RuntimeError(f"Failed to initialize DeepEP buffers: {err}")
            
    def __del__(self):
        if hasattr(self, 'params'):
            _lib.deep_ep_free_buffer(ctypes.byref(self.params))
            
    def dispatch(
        self,
        input_tensor: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """Dispatch tokens to experts"""
        assert input_tensor.is_cuda and expert_indices.is_cuda and expert_weights.is_cuda
        
        batch_size, seq_len, hidden_size = input_tensor.shape
        self.params.num_tokens = batch_size * seq_len
        self.params.hidden_dim = hidden_size
        
        # Allocate output tensor
        output = torch.zeros(
            (self.num_experts, batch_size * seq_len, hidden_size),
            dtype=input_tensor.dtype,
            device=input_tensor.device
        )
        
        with torch.cuda.stream(self.stream):
            err = _lib.deep_ep_dispatch(
                input_tensor.data_ptr(),
                expert_indices.data_ptr(),
                expert_weights.data_ptr(),
                output.data_ptr(),
                ctypes.byref(self.params)
            )
            if err != 0:
                raise RuntimeError(f"Failed to dispatch tokens to experts: {err}")
                
        return output
        
    def combine(
        self,
        expert_outputs: torch.Tensor,
        expert_indices: torch.Tensor,
        expert_weights: torch.Tensor
    ) -> torch.Tensor:
        """Combine expert outputs"""
        assert expert_outputs.is_cuda and expert_indices.is_cuda and expert_weights.is_cuda
        
        num_experts, tokens, hidden_size = expert_outputs.shape
        self.params.num_tokens = tokens
        self.params.hidden_dim = hidden_size
        
        # Allocate output tensor
        output = torch.zeros(
            (tokens, hidden_size),
            dtype=expert_outputs.dtype,
            device=expert_outputs.device
        )
        
        with torch.cuda.stream(self.stream):
            err = _lib.deep_ep_combine(
                expert_outputs.data_ptr(),
                expert_indices.data_ptr(),
                expert_weights.data_ptr(),
                output.data_ptr(),
                ctypes.byref(self.params)
            )
            if err != 0:
                raise RuntimeError(f"Failed to combine expert outputs: {err}")
                
        return output

def create_deep_ep_module(hidden_size: int, num_experts: int) -> DeepEPModule:
    """Create a DeepEP module for efficient expert parallelism"""
    return DeepEPModule(hidden_size, num_experts)
