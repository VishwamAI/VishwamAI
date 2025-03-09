"""
Performance tuning utilities for VishwamAI attention mechanisms.
Automatically profiles and selects optimal attention variants based on hardware and input characteristics.
"""

import torch
import jax
import jax.numpy as jnp
from typing import Dict, Optional, Union, Tuple
import logging
from dataclasses import dataclass
import time
import numpy as np

@dataclass
class AttentionConfig:
    """Configuration for attention mechanism selection and tuning"""
    batch_size: int
    seq_length: int
    embed_dim: int
    num_heads: int
    device_type: str  # "gpu" or "tpu"
    attention_type: str = "auto"  # "base", "flash_mla", "temporal", "multimodal" or "auto"
    block_size: Optional[int] = None
    num_domains: Optional[int] = None
    max_temporal_length: Optional[int] = None

class AttentionProfiler:
    """Profiles different attention variants to select optimal configuration"""
    
    def __init__(self, config: AttentionConfig):
        self.config = config
        self.results = {}
        
    def profile_attention_variant(self, variant_name: str, attention_cls, **kwargs) -> float:
        """Profile a specific attention variant and return execution time"""
        try:
            # Create sample inputs
            if self.config.device_type == "gpu":
                x = torch.randn(
                    self.config.batch_size,
                    self.config.seq_length,
                    self.config.embed_dim,
                    device="cuda"
                )
            else:  # TPU
                x = jnp.array(
                    np.random.randn(
                        self.config.batch_size,
                        self.config.seq_length,
                        self.config.embed_dim
                    )
                )
            
            # Initialize attention
            attention = attention_cls(
                embed_dim=self.config.embed_dim,
                num_heads=self.config.num_heads,
                **kwargs
            )
            
            # Warmup
            for _ in range(3):
                if self.config.device_type == "gpu":
                    with torch.cuda.amp.autocast():
                        _ = attention(x)
                    torch.cuda.synchronize()
                else:
                    _ = attention(x)
                    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
            
            # Profile
            times = []
            iterations = 10
            
            for _ in range(iterations):
                start = time.perf_counter()
                
                if self.config.device_type == "gpu":
                    with torch.cuda.amp.autocast():
                        _ = attention(x)
                    torch.cuda.synchronize()
                else:
                    _ = attention(x)
                    jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
                
                end = time.perf_counter()
                times.append(end - start)
            
            avg_time = np.mean(times)
            self.results[variant_name] = {
                "avg_time": avg_time,
                "std_time": np.std(times),
                "config": kwargs
            }
            return avg_time
            
        except Exception as e:
            logging.warning(f"Failed to profile {variant_name}: {str(e)}")
            return float('inf')

    def select_optimal_config(self) -> Tuple[str, Dict]:
        """Select the optimal attention configuration based on profiling results"""
        if not self.results:
            raise ValueError("No profiling results available. Run profiling first.")
            
        best_variant = min(self.results.items(), key=lambda x: x[1]["avg_time"])
        return best_variant[0], best_variant[1]["config"]

def tune_attention(config: AttentionConfig) -> Tuple[str, Dict]:
    """
    Profile and select optimal attention configuration for given parameters
    
    Args:
        config: AttentionConfig with desired parameters
        
    Returns:
        Tuple of (selected_variant_name, configuration_dict)
    """
    profiler = AttentionProfiler(config)
    
    if config.device_type == "gpu":
        from vishwamai.models.gpu.attention import (
            BaseAttention,
            FlashMLAAttention,
            MultiModalAttention,
            TemporalAttention
        )
        
        # Profile GPU variants
        profiler.profile_attention_variant("base", BaseAttention)
        
        profiler.profile_attention_variant(
            "flash_mla",
            FlashMLAAttention,
            block_size=config.block_size or 128
        )
        
        if config.num_domains:
            profiler.profile_attention_variant(
                "multimodal",
                MultiModalAttention,
                num_domains=config.num_domains
            )
            
        if config.max_temporal_length:
            profiler.profile_attention_variant(
                "temporal",
                TemporalAttention,
                max_temporal_length=config.max_temporal_length
            )
            
    else:  # TPU
        from vishwamai.models.tpu.attention import (
            BaseAttention,
            FlashMLAttentionTPU,
            MultiModalAttentionTPU,
            TemporalAttentionTPU
        )
        
        # Profile TPU variants
        profiler.profile_attention_variant("base", BaseAttention)
        
        profiler.profile_attention_variant(
            "flash_mla",
            FlashMLAttentionTPU,
            block_size=config.block_size or 128
        )
        
        if config.num_domains:
            profiler.profile_attention_variant(
                "multimodal",
                MultiModalAttentionTPU,
                num_domains=config.num_domains
            )
            
        if config.max_temporal_length:
            profiler.profile_attention_variant(
                "temporal",
                TemporalAttentionTPU,
                max_temporal_length=config.max_temporal_length
            )
    
    return profiler.select_optimal_config()

def create_optimized_attention(config: AttentionConfig):
    """
    Create an optimized attention instance based on profiling results
    
    Args:
        config: AttentionConfig with desired parameters
        
    Returns:
        Instantiated attention module with optimal configuration
    """
    variant_name, variant_config = tune_attention(config)
    
    if config.device_type == "gpu":
        from vishwamai.models.gpu.attention import (
            BaseAttention,
            FlashMLAAttention,
            MultiModalAttention,
            TemporalAttention
        )
        
        attention_classes = {
            "base": BaseAttention,
            "flash_mla": FlashMLAAttention,
            "multimodal": MultiModalAttention,
            "temporal": TemporalAttention
        }
    else:
        from vishwamai.models.tpu.attention import (
            BaseAttention,
            FlashMLAttentionTPU,
            MultiModalAttentionTPU,
            TemporalAttentionTPU
        )
        
        attention_classes = {
            "base": BaseAttention,
            "flash_mla": FlashMLAttentionTPU,
            "multimodal": MultiModalAttentionTPU,
            "temporal": TemporalAttentionTPU
        }
    
    attention_cls = attention_classes[variant_name]
    return attention_cls(
        embed_dim=config.embed_dim,
        num_heads=config.num_heads,
        **variant_config
    )