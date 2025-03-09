"""
CUDA kernel interface layer with distributed processing support via smallpond.
Provides unified interface for optimized CUDA kernels.
"""

import torch
import torch.utils.cpp_extension
import os
import numpy as np
import smallpond
from typing import Optional, Dict, Any, List, Tuple
from dataclasses import dataclass
from pathlib import Path

@dataclass
class KernelMetadata:
    """Metadata for CUDA kernel"""
    name: str
    sm_target: int  # Target SM architecture (e.g., 80 for A100)
    num_threads: int
    shared_mem: int
    block_size: Tuple[int, ...]
    grid_size: Tuple[int, ...]

class CUDAKernelManager:
    """Manages CUDA kernels with distributed execution support"""
    
    def __init__(self,
                use_smallpond: bool = True,
                cache_dir: Optional[str] = "/tmp/vishwamai/cuda_kernels"):
        self.cache_dir = cache_dir
        self.use_smallpond = use_smallpond
        os.makedirs(cache_dir, exist_ok=True)
        
        # Load CUDA kernels
        self.kernel_dir = Path(__file__).parent / "kernels"
        self.kernels = {}
        self._load_kernels()
        
        # Initialize smallpond
        if use_smallpond:
            try:
                self.sp_session = smallpond.init(
                    num_executors=torch.cuda.device_count(),
                    data_root=cache_dir,
                    bind_numa_node=True
                )
            except:
                self.sp_session = None
                self.use_smallpond = False
                
        # Performance tracking
        self.perf_stats = {}
        
    def _load_kernels(self):
        """Load CUDA kernels from source files"""
        # DeepGEMM kernels
        self._load_kernel_group("deepgemm", [
            "gemm_fp8_kernel.cu",
            "gemm_bf16_kernel.cu"
        ])
        
        # Flash MLA kernels
        self._load_kernel_group("flash_mla", [
            "flash_mla_fwd.cu",
            "flash_mla_bwd.cu"
        ])
        
        # Expert parallelism kernels
        self._load_kernel_group("deep_ep", [
            "dispatch.cu",
            "combine.cu"
        ])
        
    def _load_kernel_group(self, group: str, files: List[str]):
        """Load a group of related kernels"""
        group_dir = self.kernel_dir / group
        
        for file in files:
            name = Path(file).stem
            path = group_dir / file
            
            if not path.exists():
                continue
                
            try:
                module = torch.utils.cpp_extension.load(
                    name=f"{group}_{name}",
                    sources=[str(path)],
                    verbose=False
                )
                self.kernels[f"{group}/{name}"] = module
            except Exception as e:
                print(f"Failed to load kernel {file}: {e}")
                
    def get_kernel(self, kernel_path: str) -> Any:
        """Get loaded CUDA kernel by path"""
        return self.kernels.get(kernel_path)
        
    def run_kernel_distributed(self,
                            kernel_path: str,
                            *args,
                            grid: Tuple[int, ...],
                            block: Tuple[int, ...],
                            shared_mem: int = 0,
                            stream: Optional[torch.cuda.Stream] = None):
        """Run CUDA kernel with distributed processing"""
        kernel = self.get_kernel(kernel_path)
        if kernel is None:
            raise ValueError(f"Kernel {kernel_path} not found")
            
        if not self.use_smallpond or self.sp_session is None:
            # Direct execution
            kernel[grid, block, stream, shared_mem](*args)
            return
            
        # Create input dataframe for distribution
        input_arrays = []
        for arg in args:
            if isinstance(arg, torch.Tensor):
                input_arrays.append(arg.detach().cpu().numpy())
            else:
                input_arrays.append(arg)
                
        df = self.sp_session.create_dataframe({
            'inputs': [input_arrays],
            'grid': [grid],
            'block': [block],
            'shared_mem': [shared_mem]
        })
        
        # Partition work
        num_splits = min(
            self.sp_session.num_executors,
            grid[0]  # Split along first grid dimension
        )
        df = df.repartition(num_splits)
        
        # Process partitions
        def process_partition(partition):
            import torch
            import numpy as np
            
            # Convert inputs back to tensors/types
            inputs = []
            for arr in partition['inputs'].iloc[0]:
                if isinstance(arr, np.ndarray):
                    inputs.append(torch.from_numpy(arr))
                else:
                    inputs.append(arr)
                    
            # Adjust grid size for partition
            grid = list(partition['grid'].iloc[0])
            grid[0] = grid[0] // num_splits
            
            # Run kernel
            kernel[
                tuple(grid),
                tuple(partition['block'].iloc[0]),
                stream,
                partition['shared_mem'].iloc[0]
            ](*inputs)
            
            # Return any modified tensors
            outputs = []
            for inp in inputs:
                if isinstance(inp, torch.Tensor):
                    outputs.append(inp.cpu().numpy())
                else:
                    outputs.append(inp)
            return outputs
            
        # Execute and gather results
        result_df = df.map_partitions(process_partition)
        
        # Update performance stats
        kernel_name = kernel_path.split('/')[-1]
        if kernel_name not in self.perf_stats:
            self.perf_stats[kernel_name] = {
                'calls': 0,
                'total_time': 0.0
            }
        self.perf_stats[kernel_name]['calls'] += 1
        
    def get_performance_stats(self) -> Dict[str, Dict[str, float]]:
        """Get kernel performance statistics"""
        return self.perf_stats
        
    def cleanup(self):
        """Cleanup resources"""
        if self.use_smallpond and self.sp_session:
            self.sp_session.shutdown()

# Global kernel manager instance
_kernel_manager = None

def get_kernel_manager() -> CUDAKernelManager:
    """Get or create global kernel manager instance"""
    global _kernel_manager
    if _kernel_manager is None:
        _kernel_manager = CUDAKernelManager()
    return _kernel_manager

def cleanup_kernels():
    """Cleanup global kernel manager"""
    global _kernel_manager
    if _kernel_manager is not None:
        _kernel_manager.cleanup()
        _kernel_manager = None