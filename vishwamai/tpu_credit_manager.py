"""TPU credit usage management and optimization."""

import time
import json
import logging
from dataclasses import dataclass
from typing import Dict, Any, Optional
import jax
import jax.numpy as jnp

@dataclass
class TPUCreditConfig:
    """Configuration for TPU credit management."""
    max_credits: float = 113371.0  # Total available credits
    warning_threshold: float = 0.8  # 80% of credits
    critical_threshold: float = 0.9  # 90% of credits
    cost_per_core_hour: float = 0.35  # TPU v3 cost per core hour
    num_cores: int = 8

class TPUCreditManager:
    """Manages and optimizes TPU credit usage."""
    
    def __init__(self, config: TPUCreditConfig):
        self.config = config
        self.start_time = time.time()
        self.credits_used = 0.0
        self.last_update = self.start_time
        self.usage_history = []
        
        # Initialize metrics
        self.metrics = {
            "total_compute_hours": 0.0,
            "total_memory_hours": 0.0,
            "avg_utilization": 0.0,
            "credits_remaining": config.max_credits
        }
        
        self.logger = logging.getLogger("TPUCreditManager")
    
    def update_usage(self, step_metrics: Optional[Dict[str, float]] = None):
        """Update credit usage based on elapsed time and metrics."""
        current_time = time.time()
        hours_elapsed = (current_time - self.last_update) / 3600
        
        # Calculate core hours
        core_hours = hours_elapsed * self.config.num_cores
        cost = core_hours * self.config.cost_per_core_hour
        
        # Update credits
        self.credits_used += cost
        self.metrics["credits_remaining"] = self.config.max_credits - self.credits_used
        
        # Update compute metrics
        self.metrics["total_compute_hours"] += core_hours
        self.metrics["total_memory_hours"] += core_hours * 8  # 8GB per core
        
        # Update utilization if metrics provided
        if step_metrics:
            utilization = step_metrics.get("compute_util", 0.0)
            self.metrics["avg_utilization"] = (
                self.metrics["avg_utilization"] * 0.95 + utilization * 0.05
            )
        
        # Record usage
        self.usage_history.append({
            "timestamp": current_time,
            "credits_used": cost,
            "total_credits": self.credits_used,
            "utilization": self.metrics["avg_utilization"]
        })
        
        self.last_update = current_time
        
        # Check thresholds
        self._check_credit_thresholds()
    
    def get_optimal_batch_size(self, current_batch_size: int) -> int:
        """Dynamically adjust batch size based on credit usage and performance."""
        if self.credits_used >= self.config.max_credits * self.config.warning_threshold:
            # Reduce batch size to conserve credits
            return max(1, current_batch_size // 2)
        
        if self.metrics["avg_utilization"] < 0.7:
            # Increase batch size if utilization is low
            return min(256, current_batch_size * 2)
            
        return current_batch_size
    
    def should_enable_optimizations(self) -> Dict[str, bool]:
        """Determine which optimizations to enable based on credit usage."""
        credit_ratio = self.credits_used / self.config.max_credits
        
        return {
            "use_gradient_checkpointing": credit_ratio > 0.5,
            "use_fp8": credit_ratio > 0.6,
            "use_model_parallel": credit_ratio > 0.7,
            "reduce_precision": credit_ratio > 0.8
        }
    
    def get_training_recommendation(self) -> Dict[str, Any]:
        """Get recommendations for training configuration."""
        credit_ratio = self.credits_used / self.config.max_credits
        
        if credit_ratio < 0.3:
            return {
                "mode": "fast",
                "batch_size_multiplier": 2.0,
                "gradient_accumulation": 1
            }
        elif credit_ratio < 0.6:
            return {
                "mode": "balanced",
                "batch_size_multiplier": 1.0,
                "gradient_accumulation": 2
            }
        else:
            return {
                "mode": "efficient",
                "batch_size_multiplier": 0.5,
                "gradient_accumulation": 4
            }
    
    def _check_credit_thresholds(self):
        """Check credit usage against thresholds and log warnings."""
        credit_ratio = self.credits_used / self.config.max_credits
        
        if credit_ratio >= self.config.critical_threshold:
            self.logger.warning(
                f"CRITICAL: {credit_ratio:.1%} of credits used! Consider reducing batch size "
                "or enabling more aggressive optimizations."
            )
        elif credit_ratio >= self.config.warning_threshold:
            self.logger.warning(
                f"WARNING: {credit_ratio:.1%} of credits used. Enabling credit-saving optimizations."
            )
    
    def save_usage_stats(self, filepath: str):
        """Save credit usage statistics to file."""
        with open(filepath, 'w') as f:
            json.dump({
                "credits_used": self.credits_used,
                "credits_remaining": self.metrics["credits_remaining"],
                "compute_hours": self.metrics["total_compute_hours"],
                "memory_hours": self.metrics["total_memory_hours"],
                "avg_utilization": self.metrics["avg_utilization"],
                "usage_history": self.usage_history
            }, f, indent=2)
    
    def estimate_remaining_training_time(self, 
                                      steps_per_hour: float, 
                                      remaining_steps: int) -> float:
        """Estimate remaining training time possible with current credits."""
        remaining_credits = self.metrics["credits_remaining"]
        cost_per_hour = self.config.num_cores * self.config.cost_per_core_hour
        remaining_hours = remaining_credits / cost_per_hour
        
        possible_steps = remaining_hours * steps_per_hour
        return min(possible_steps, float(remaining_steps))