"""
DeepGEMM: Optimized GEMM operations for GPU with distributed processing via smallpond
"""

import torch
import os
import math
import time
import smallpond
import numpy as np
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass

@dataclass
class GEMMConfig:
    """Configuration for DeepGEMM operations"""
    block_m: int = 128
    block_n: int = 128
    block_k: int = 32
    num_stages: int = 3
    warps_m: int = 4
    warps_n: int = 4
    use_fp8: bool = True
    use_async: bool = True

class DistributedGEMM:
    """Distributed GEMM computation using smallpond"""
    
    def __init__(self,
                num_executors: Optional[int] = None,
                cache_dir: Optional[str] = "/tmp/vishwamai/deepgemm_cache",
                config: Optional[GEMMConfig] = None):
        self.cache_dir = cache_dir
        self.config = config or GEMMConfig()
        os.makedirs(cache_dir, exist_ok=True)
        
        # Initialize smallpond
        try:
            self.sp_session = smallpond.init(
                num_executors=num_executors or torch.cuda.device_count(),
                data_root=cache_dir,
                bind_numa_node=True
            )
        except:
            self.sp_session = None
            
        # Performance tracking
        self.perf_stats = {
            'total_compute_time': 0.0,
            'num_operations': 0,
            'avg_throughput': 0.0
        }
        
    def distribute_gemm(self,
                      a: torch.Tensor,
                      b: torch.Tensor,
                      bias: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Distribute GEMM computation across executors"""
        start_time = time.time()  # Define start_time at the beginning of the method
        
        if self.sp_session is None:
            return gemm_fp8_fp8_bf16_nt(a, b, bias, self.config)
            
        # Convert to numpy
        a_np = a.detach().cpu().numpy()
        b_np = b.detach().cpu().numpy()
        bias_np = bias.detach().cpu().numpy() if bias is not None else None
        
        # Create dataframe with inputs
        df = self.sp_session.create_dataframe({
            'a': [a_np],
            'b': [b_np],
            'bias': [bias_np] if bias_np is not None else None
        })
        
        # Partition based on matrix size
        num_splits = min(
            self.sp_session.num_executors,
            math.ceil(a.size(0) / self.config.block_m)
        )
        df = df.repartition(num_splits)
        
        # Process partitions
        def process_partition(partition):
            import torch
            import numpy as np
            
            # Convert inputs back to tensors
            a = torch.from_numpy(np.array(partition['a'].iloc[0]))
            b = torch.from_numpy(np.array(partition['b'].iloc[0]))
            bias = (
                torch.from_numpy(np.array(partition['bias'].iloc[0]))
                if 'bias' in partition.columns else None
            )
            
            # Run GEMM kernel
            output = gemm_fp8_fp8_bf16_nt(a, b, bias, self.config)
            return output.cpu().numpy()
            
        result_df = df.map_partitions(process_partition)
        
        # Gather and combine results
        results = []
        for _, row in result_df.to_pandas().iterrows():
            results.append(row['data'])
        result = torch.from_numpy(np.concatenate(results))
        
        # Update stats
        compute_time = time.time() - start_time
        self.perf_stats['total_compute_time'] += compute_time
        self.perf_stats['num_operations'] += 1
        self.perf_stats['avg_throughput'] = (
            self.perf_stats['num_operations'] / 
            self.perf_stats['total_compute_time']
        )
        
        return result.to(a.device)
        
    def cleanup(self):
        """Cleanup resources"""
        if self.sp_session:
            self.sp_session.shutdown()

def gemm_fp8_fp8_bf16_nt(a: torch.Tensor,
                         b: torch.Tensor,
                         bias: Optional[torch.Tensor] = None,
                         config: Optional[GEMMConfig] = None) -> torch.Tensor:
    """
    Optimized GEMM computation with FP8/BF16 mixed precision
    
    Args:
        a: Input tensor A
        b: Input tensor B
        bias: Optional bias tensor
        config: GEMM configuration
    
    Returns:
        Output tensor
    """
    if not config:
        config = GEMMConfig()
        
    # Set optimal configuration
    set_warps(config.warps_m, config.warps_n)
    
    # Get optimal kernel
    kernel_fn = get_best_kernel(
        a.shape,
        b.shape,
        use_fp8=config.use_fp8
    )
    
    # Run computation
    output = kernel_fn(
        a, b,
        block_m=config.block_m,
        block_n=config.block_n,
        block_k=config.block_k,
        num_stages=config.num_stages
    )
    
    if bias is not None:
        output += bias
        
    return output

def get_best_kernel(a_shape: Tuple[int, ...],
                   b_shape: Tuple[int, ...],
                   use_fp8: bool = True):
    """Get optimal GEMM kernel for input shapes"""
    from .kernels import (
        gemm_fp8_kernel,
        gemm_bf16_kernel
    )
    
    if use_fp8 and torch.cuda.get_device_capability()[0] >= 8:
        return gemm_fp8_kernel
    return gemm_bf16_kernel

def get_best_configs(hidden_size: int,
                    seq_len: int,
                    batch_size: int) -> GEMMConfig:
    """Get optimal GEMM configuration for problem size"""
    # Tune block sizes based on matrix dimensions
    block_m = min(128, seq_len)
    block_n = min(128, hidden_size)
    block_k = min(32, hidden_size)
    
    # Scale warps based on GPU capability
    num_sms = torch.cuda.get_device_properties(0).multi_processor_count
    warps_m = min(4, num_sms // 4)
    warps_n = min(4, num_sms // 4)
    
    return GEMMConfig(
        block_m=block_m,
        block_n=block_n,
        block_k=block_k,
        warps_m=warps_m,
        warps_n=warps_n,
        use_fp8=torch.cuda.get_device_capability()[0] >= 8
    )

def layernorm(x: torch.Tensor,
             weight: Optional[torch.Tensor] = None,
             bias: Optional[torch.Tensor] = None,
             eps: float = 1e-5) -> torch.Tensor:
    """Optimized layer normalization"""
    mean = x.mean(-1, keepdim=True)
    variance = x.var(-1, keepdim=True, unbiased=False)
    norm = (x - mean) / torch.sqrt(variance + eps)
    
    if weight is not None:
        norm = norm * weight
    if bias is not None:
        norm = norm + bias
        
    return norm

def get_num_sms() -> int:
    """Get number of SMs on GPU"""
    if not torch.cuda.is_available():
        return 1
    return torch.cuda.get_device_properties(0).multi_processor_count

def set_warps(warps_m: int, warps_n: int):
    """Set number of warps for GEMM kernels"""
    from .kernels import set_kernel_warps
    set_kernel_warps(warps_m, warps_n)

def init_deepgemm_kernels():
    """Initialize DeepGEMM CUDA kernels"""
    from .kernels import init_kernels
    init_kernels()

# Initialize kernels on import
init_deepgemm_kernels()
