"""Profiling utilities for performance monitoring."""
from typing import Any, Dict, List, Optional, Union, Callable
import time
import torch
import numpy as np
from functools import wraps
from contextlib import ContextDecorator

class Timer(ContextDecorator):
    """Timer context manager/decorator for performance measurement.
    
    Args:
        name: Name of operation being timed
        logger: Optional logger to record times
    """
    
    def __init__(
        self,
        name: str,
        logger: Optional[Any] = None
    ):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.times = []
        
    def __enter__(self):
        """Start timer."""
        self.start_time = time.perf_counter()
        return self
        
    def __exit__(self, *exc):
        """Stop timer and log."""
        elapsed = time.perf_counter() - self.start_time
        self.times.append(elapsed)
        
        if self.logger:
            self.logger.log_metrics(
                {f"time/{self.name}": elapsed},
                step=len(self.times)
            )
            
        return False
        
    @property
    def mean_time(self) -> float:
        """Get mean execution time."""
        return np.mean(self.times)
        
    @property
    def std_time(self) -> float:
        """Get standard deviation of execution time."""
        return np.std(self.times)

class MemoryTracker:
    """Track CUDA memory usage.
    
    Args:
        logger: Optional logger to record stats
        log_every: How often to log stats
    """
    
    def __init__(
        self,
        logger: Optional[Any] = None,
        log_every: int = 100
    ):
        self.logger = logger
        self.log_every = log_every
        self.step = 0
        self.peak_allocated = 0
        self.peak_cached = 0
        
    def track(self) -> None:
        """Record current memory usage."""
        if not torch.cuda.is_available():
            return
            
        self.step += 1
        allocated = torch.cuda.max_memory_allocated()
        cached = torch.cuda.max_memory_reserved()
        
        self.peak_allocated = max(self.peak_allocated, allocated)
        self.peak_cached = max(self.peak_cached, cached)
        
        if self.logger and self.step % self.log_every == 0:
            self.logger.log_metrics({
                'memory/allocated_gb': allocated / 1e9,
                'memory/cached_gb': cached / 1e9,
                'memory/peak_allocated_gb': self.peak_allocated / 1e9,
                'memory/peak_cached_gb': self.peak_cached / 1e9
            }, self.step)
            
    def reset(self) -> None:
        """Reset peak stats."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        self.peak_allocated = 0
        self.peak_cached = 0

class ExpertProfiler:
    """Profile MoE expert usage and performance.
    
    Args:
        num_experts: Number of experts
        logger: Optional logger to record stats
        log_every: How often to log stats
    """
    
    def __init__(
        self,
        num_experts: int,
        logger: Optional[Any] = None,
        log_every: int = 100
    ):
        self.num_experts = num_experts
        self.logger = logger
        self.log_every = log_every
        
        self.step = 0
        self.expert_calls = np.zeros(num_experts)
        self.expert_times = [[] for _ in range(num_experts)]
        self.token_counts = np.zeros(num_experts)
        
    def log_expert_call(
        self,
        expert_idx: int,
        time_taken: float,
        num_tokens: int
    ) -> None:
        """Log single expert call.
        
        Args:
            expert_idx: Index of called expert
            time_taken: Time taken for forward pass
            num_tokens: Number of tokens processed
        """
        self.expert_calls[expert_idx] += 1
        self.expert_times[expert_idx].append(time_taken)
        self.token_counts[expert_idx] += num_tokens
        
        self.step += 1
        if self.logger and self.step % self.log_every == 0:
            self._log_stats()
            
    def _log_stats(self) -> None:
        """Log expert statistics."""
        # Calculate usage stats
        total_calls = self.expert_calls.sum()
        usage = self.expert_calls / max(1, total_calls)
        
        # Calculate timing stats
        mean_times = np.array([
            np.mean(times) if times else 0
            for times in self.expert_times
        ])
        std_times = np.array([
            np.std(times) if times else 0
            for times in self.expert_times
        ])
        
        # Calculate token stats
        total_tokens = self.token_counts.sum()
        token_fracs = self.token_counts / max(1, total_tokens)
        
        # Log stats
        self.logger.log_expert_stats({
            'expert_usage': usage.tolist(),
            'expert_mean_time': mean_times.tolist(),
            'expert_std_time': std_times.tolist(),
            'expert_token_frac': token_fracs.tolist(),
            'expert_calls': self.expert_calls.tolist(),
            'expert_tokens': self.token_counts.tolist()
        }, self.step)
        
    def reset(self) -> None:
        """Reset profiling stats."""
        self.expert_calls = np.zeros(self.num_experts)
        self.expert_times = [[] for _ in range(self.num_experts)]
        self.token_counts = np.zeros(self.num_experts)

def profile_cuda(func: Callable) -> Callable:
    """Decorator to profile CUDA kernels in a function.
    
    Args:
        func: Function to profile
        
    Returns:
        Wrapped function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            return func(*args, **kwargs)
            
        # Start profiling
        torch.cuda.synchronize()
        with torch.cuda.profiler.profile() as prof:
            with torch.cuda.nvtx.range(func.__name__):
                result = func(*args, **kwargs)
        torch.cuda.synchronize()
        
        # Print profile
        print(f"\nCUDA Profile for {func.__name__}:")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=10
        ))
        
        return result
    return wrapper

def print_gpu_utilization():
    """Print current GPU memory usage."""
    if not torch.cuda.is_available():
        return
        
    print("GPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.2f}GB")
    print(f"Cached: {torch.cuda.memory_reserved() / 1e9:.2f}GB")
    print(f"Peak Allocated: {torch.cuda.max_memory_allocated() / 1e9:.2f}GB")
    print(f"Peak Cached: {torch.cuda.max_memory_reserved() / 1e9:.2f}GB")
