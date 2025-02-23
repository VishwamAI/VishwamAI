"""Performance profiling utilities for model training."""

import time
from typing import Dict, List, Optional
import numpy as np
import torch
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)

class PerformanceProfiler:
    """Profiles model training performance metrics."""
    
    def __init__(self):
        self.step_times: List[float] = []
        self.forward_times: List[float] = []
        self.backward_times: List[float] = []
        self.optimization_times: List[float] = []
        self.throughput_records: List[float] = []
        self._start_time: Optional[float] = None
        self._temp_timings = defaultdict(float)
        
    def start_event(self, event_name: str) -> None:
        """Start timing an event.
        
        Args:
            event_name (str): Name of the event to time
        """
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        try:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        except ImportError:
            pass
        self._temp_timings[f"{event_name}_start"] = time.perf_counter()
        
    def end_event(self, event_name: str) -> float:
        """End timing an event and return duration.
        
        Args:
            event_name (str): Name of the event to end timing
            
        Returns:
            float: Duration of the event in seconds
        """
        torch.cuda.synchronize() if torch.cuda.is_available() else None
        try:
            import torch_xla.core.xla_model as xm
            xm.mark_step()
        except ImportError:
            pass
            
        end_time = time.perf_counter()
        start_time = self._temp_timings.pop(f"{event_name}_start", None)
        
        if start_time is None:
            logger.warning(f"No start time found for event {event_name}")
            return 0.0
            
        duration = end_time - start_time
        
        if event_name == "step":
            self.step_times.append(duration)
        elif event_name == "forward":
            self.forward_times.append(duration)
        elif event_name == "backward":
            self.backward_times.append(duration)
        elif event_name == "optimization":
            self.optimization_times.append(duration)
            
        return duration
        
    def record_throughput(self, num_samples: int, duration: float) -> None:
        """Record throughput for a batch of samples.
        
        Args:
            num_samples (int): Number of samples processed
            duration (float): Time taken to process samples in seconds
        """
        throughput = num_samples / duration
        self.throughput_records.append(throughput)
        
    def get_performance_stats(self) -> Dict[str, float]:
        """Get accumulated performance statistics.
        
        Returns:
            dict: Dictionary containing performance statistics
        """
        stats = {}
        
        if self.step_times:
            stats.update({
                'avg_step_time': np.mean(self.step_times),
                'std_step_time': np.std(self.step_times),
                'min_step_time': np.min(self.step_times),
                'max_step_time': np.max(self.step_times)
            })
            
        if self.forward_times:
            stats['avg_forward_time'] = np.mean(self.forward_times)
            
        if self.backward_times:
            stats['avg_backward_time'] = np.mean(self.backward_times)
            
        if self.optimization_times:
            stats['avg_optimization_time'] = np.mean(self.optimization_times)
            
        if self.throughput_records:
            stats.update({
                'avg_throughput': np.mean(self.throughput_records),
                'peak_throughput': np.max(self.throughput_records)
            })
            
        return stats
        
    def log_performance_stats(self) -> None:
        """Log current performance statistics."""
        stats = self.get_performance_stats()
        
        for name, value in stats.items():
            if 'time' in name:
                logger.info(f"{name}: {value:.4f} seconds")
            elif 'throughput' in name:
                logger.info(f"{name}: {value:.2f} samples/second")
            else:
                logger.info(f"{name}: {value:.4f}")
                
    def reset(self) -> None:
        """Reset all performance tracking metrics."""
        self.step_times.clear()
        self.forward_times.clear()
        self.backward_times.clear()
        self.optimization_times.clear()
        self.throughput_records.clear()
        self._temp_timings.clear()
