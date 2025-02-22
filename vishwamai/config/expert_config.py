"""
Configuration for Mixture of Experts and parallel components
"""
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Literal, Dict, Any
import warnings
import torch

try:
    import flash_attn
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False

try:
    import xformers
    HAS_XFORMERS = True
except ImportError:
    HAS_XFORMERS = False

@dataclass
class ExpertConfig:
    """Configuration for Mixture of Experts (MoE)"""
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity: int = 32
    expert_dtype: Optional[str] = None
    expert_group_size: int = 1
    expert_placement: Literal["balanced", "dense", "sparse"] = "balanced"
    gating_type: Literal["top_k", "hash", "random"] = "top_k"
    load_balancing_weight: float = 0.01
    aux_loss_weight: float = 0.01
    z_loss_weight: float = 0.0001
    expert_parallel: bool = True
    shard_experts: bool = True
    expert_activation: Literal["gelu", "swiglu", "geglu"] = "swiglu"

@dataclass
class ParallelConfig:
    """Configuration for parallel processing"""
    sequence_parallel: bool = False
    use_sequence_parallelism: bool = True
    pipeline_parallel_size: int = 1
    tensor_parallel_size: int = 1
    expert_parallel_size: int = 1
    parallel_mode: Literal["naive", "tensor", "pipeline", "expert"] = "tensor"
    reduce_scatter: bool = True
    gradient_sync_freq: int = 1
    communication_dtype: str = "fp32"
    overlap_computation: bool = True
    use_kernel_optimizations: bool = True

@dataclass
class AdvancedMLPConfig:
    """Configuration for advanced MLP components"""
    mlp_ratio: float = 4.0
    activation_fn: str = "gelu"
    gated_mlp: bool = True
    up_proj_fn: str = "linear"
    down_proj_fn: str = "linear"
    gate_up_fn: str = "linear"
    use_bias: bool = True
    intermediate_size_ratio: float = 1.0
    use_expert_choice: bool = True
    expert_capacity_factor: float = 1.0

@dataclass
class AdvancedTransformerConfig:
    """Configuration for advanced transformer components"""
    use_rms_norm: bool = True
    norm_eps: float = 1e-6
    prenorm: bool = True
    use_alibi: bool = False
    use_gated_mlp: bool = True
    use_parallel_attention: bool = True
    use_memory_efficient_attention: bool = True
    use_xformers: bool = True
    attention_softmax_in_fp32: bool = True
    scale_attn_by_inverse_layer_idx: bool = True
    residual_in_fp32: bool = True
    use_flash_attention: bool = True
    fused_qkv: bool = True
    fused_mlp: bool = True
    fused_dropout_add_ln: bool = True
    rotary_base: int = 10000
    rotary_scaling: Optional[Dict[str, Any]] = None

@dataclass
class ParallelLinearConfig:
    """Configuration for parallel linear layers"""
    init_method: str = "xavier_uniform"
    output_layer_init_method: str = "xavier_uniform"
    bias: bool = True
    gather_output: bool = True
    params_dtype: str = "fp32"
    sequence_parallel: bool = False
    gradient_accumulation_fusion: bool = True
    cpu_offload: bool = False
    triton_backend: bool = True
    async_communication: bool = True
    reduce_scatter: bool = True

@dataclass
class UnifiedConfig:
    """Unified configuration for all advanced components"""
    expert: ExpertConfig = field(default_factory=ExpertConfig)
    parallel: ParallelConfig = field(default_factory=ParallelConfig)
    mlp: AdvancedMLPConfig = field(default_factory=AdvancedMLPConfig)
    transformer: AdvancedTransformerConfig = field(default_factory=AdvancedTransformerConfig)
    linear: ParallelLinearConfig = field(default_factory=ParallelLinearConfig)

    def validate(self):
        """Validate configuration settings"""
        if self.expert.num_experts > 0:
            assert self.expert.num_experts_per_token > 0, \
                "num_experts_per_token must be positive when using experts"
            assert self.expert.expert_capacity > 0, \
                "expert_capacity must be positive when using experts"
            
        if self.parallel.tensor_parallel_size > 1:
            assert self.parallel.tensor_parallel_size <= torch.cuda.device_count(), \
                "tensor_parallel_size cannot exceed available GPU count"
                
        if self.transformer.use_flash_attention and not HAS_FLASH_ATTN:
            warnings.warn("flash_attn not available, falling back to standard attention")
            self.transformer.use_flash_attention = False

        if self.transformer.use_xformers and not HAS_XFORMERS:
            warnings.warn("xformers not available, falling back to standard attention")
            self.transformer.use_xformers = False
                
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "expert": asdict(self.expert),
            "parallel": asdict(self.parallel),
            "mlp": asdict(self.mlp),
            "transformer": asdict(self.transformer),
            "linear": asdict(self.linear)
        }
        
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'UnifiedConfig':
        """Create configuration from dictionary"""
        expert_config = ExpertConfig(**config_dict.get("expert", {}))
        parallel_config = ParallelConfig(**config_dict.get("parallel", {}))
        mlp_config = AdvancedMLPConfig(**config_dict.get("mlp", {}))
        transformer_config = AdvancedTransformerConfig(**config_dict.get("transformer", {}))
        linear_config = ParallelLinearConfig(**config_dict.get("linear", {}))
        
        return cls(
            expert=expert_config,
            parallel=parallel_config,
            mlp=mlp_config,
            transformer=transformer_config,
            linear=linear_config
        )

    def update(self, **kwargs) -> 'UnifiedConfig':
        """Update configuration with new values"""
        config_dict = self.to_dict()
        for key, value in kwargs.items():
            if "." in key:
                section, param = key.split(".", 1)
                if section in config_dict:
                    config_dict[section][param] = value
            else:
                for section_dict in config_dict.values():
                    if key in section_dict:
                        section_dict[key] = value
                        
        return self.from_dict(config_dict)

# Preset configurations
UNIFIED_BASE = UnifiedConfig(
    expert=ExpertConfig(num_experts=0),
    parallel=ParallelConfig(tensor_parallel_size=1),
    transformer=AdvancedTransformerConfig(use_flash_attention=True)
)

UNIFIED_EXPERT = UnifiedConfig(
    expert=ExpertConfig(
        num_experts=8,
        num_experts_per_token=2,
        expert_capacity=32,
        expert_parallel=True
    ),
    parallel=ParallelConfig(
        tensor_parallel_size=1,
        expert_parallel_size=8
    ),
    transformer=AdvancedTransformerConfig(
        use_flash_attention=True,
        use_gated_mlp=True
    )
)

UNIFIED_PARALLEL = UnifiedConfig(
    expert=ExpertConfig(num_experts=0),
    parallel=ParallelConfig(
        tensor_parallel_size=8,
        sequence_parallel=True,
        pipeline_parallel_size=1
    ),
    transformer=AdvancedTransformerConfig(
        use_parallel_attention=True,
        use_flash_attention=True
    )
)
