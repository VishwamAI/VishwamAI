import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass
from .base_layers import Linear
from .config import ModelArgs
from .tokenizer import VishwamAITokenizer, TokenizerConfig
from .Transformer import Transformer
from .MoE import MoE
from .hardware_adapter import HardwareAdapter, HardwareConfig
from .open_ended_learning import OpenEndedLearning, OpenEndedConfig
from .emergent_behavior import EmergentBehaviorModule, EmergentConfig
from .integrated_information import IntegratedInformationModule, IntegrationConfig
from .ethical_framework import EthicalFramework, EthicalConfig

@dataclass
class AdvancedModelConfig:
    """Configuration for advanced model components."""
    hidden_dim: int = 4096  # Increased for 7B model scale
    emergent_config: Optional[EmergentConfig] = None
    integration_config: Optional[IntegrationConfig] = None
    ethical_config: Optional[EthicalConfig] = None
    hardware_config: Optional[HardwareConfig] = None
    open_ended_config: Optional[OpenEndedConfig] = None
    tokenizer_config: Optional[TokenizerConfig] = None

def create_model(config: dict) -> nn.Module:
    """
    Create a model instance based on configuration.
    """
    is_moe = config.get("model_type") == "moe"
    
    model_args = ModelArgs(
        dim=config["hidden_size"],
        n_layers=config["num_hidden_layers"],
        n_heads=config["num_attention_heads"],
        vocab_size=config.get("vocab_size", 102400),
        n_dense_layers=0 if is_moe else config["num_hidden_layers"],
        inter_dim=config["intermediate_size"],
        max_seq_len=config.get("max_position_embeddings", 32768),  # Extended for 7B
        
        # MoE specific parameters
        n_routed_experts=config.get("num_experts", 128) if is_moe else 0,
        n_shared_experts=config.get("num_shared_experts", 4) if is_moe else 0,
        n_activated_experts=config.get("num_activated_experts", 8) if is_moe else 0,
        n_expert_groups=config.get("num_expert_groups", 2),
        moe_inter_dim=config.get("moe_intermediate_size", 2048),
        
        # Advanced features
        dtype="fp8" if config.get("mixed_precision") == "fp8" else "bf16",
        gradient_checkpointing=config.get("gradient_checkpointing", True),
        use_alibi=config.get("use_alibi", True),
        use_rope_scaling=config.get("use_rope_scaling", True),
        parallel_attn=True,
        
        # Attention dimensions
        qk_nope_head_dim=config.get("qk_nope_head_dim", 256),
        qk_rope_head_dim=config.get("qk_rope_head_dim", 128),
        v_head_dim=config.get("v_head_dim", 256),
        
        # Scaling parameters
        rope_theta=config.get("rope_theta", 20000.0),
        rope_factor=config.get("rope_factor", 80),
        mscale=config.get("mscale", 1.5),
        rope_condense_ratio=config.get("rope_condense_ratio", 1.2)
    )
    
    model = Transformer(model_args)
    
    if is_moe:
        model = MoE(model)
    
    tokenizer = VishwamAITokenizer(TokenizerConfig(
        vocab_size=config.get("vocab_size", 102400),
        max_sentence_length=config.get("max_position_embeddings", 32768)
    ))
    
    advanced_config = AdvancedModelConfig(
        hidden_dim=config["hidden_size"],
        tokenizer_config=tokenizer.config
    )
    
    components = ModelFactory.create_advanced_components(model, advanced_config)
    
    if not ModelFactory.verify_compatibility(components):
        raise ValueError("Component hidden dimensions are not compatible")
    
    return model, tokenizer

class ModelFactory:
    """Factory for creating and initializing advanced model components."""
    
    @staticmethod
    def create_advanced_components(base_model: nn.Module, config: AdvancedModelConfig) -> Dict[str, nn.Module]:
        components = {}
        
        components['emergent'] = EmergentBehaviorModule(config.hidden_dim, config.emergent_config)
        components['integration'] = IntegratedInformationModule(config.hidden_dim, config.integration_config)
        components['ethical'] = EthicalFramework(config.hidden_dim, config.ethical_config)
        components['hardware'] = HardwareAdapter(base_model, config.hardware_config)
        components['open_ended'] = OpenEndedLearning(config.hidden_dim, config.open_ended_config)
        
        return components
    
    @staticmethod
    def initialize_components(components: Dict[str, nn.Module], device: torch.device):
        for component in components.values():
            component.to(device)
            if hasattr(component, 'reset_tracking'):
                component.reset_tracking()
            if hasattr(component, 'optimize_for_hardware'):
                component.optimize_for_hardware(torch.randn(1, component.hidden_dim).to(device))
    
    @staticmethod
    def verify_compatibility(components: Dict[str, nn.Module]) -> bool:
        hidden_dims = {component.hidden_dim for component in components.values() if hasattr(component, 'hidden_dim')}
        return len(hidden_dims) == 1
    
    @staticmethod
    def get_component_stats(components: Dict[str, nn.Module]) -> Dict[str, Dict]:
        stats = {}
        for name, component in components.items():
            if hasattr(component, 'get_evolution_metrics'):
                stats[f'{name}_evolution'] = component.get_evolution_metrics()
            if hasattr(component, 'get_ethical_metrics'):
                stats[f'{name}_ethical'] = component.get_ethical_metrics()
            if hasattr(component, 'get_awareness_metrics'):
                stats[f'{name}_awareness'] = component.get_awareness_metrics()
        return stats
    
    @staticmethod
    def save_components(components: Dict[str, nn.Module], path: str):
        torch.save({name: component.state_dict() for name, component in components.items()}, path)
    
    @staticmethod
    def load_components(components: Dict[str, nn.Module], path: str):
        component_states = torch.load(path)
        for name, state in component_states.items():
            if name in components:
                components[name].load_state_dict(state)
    
    @staticmethod
    def reset_components(components: Dict[str, nn.Module]):
        for component in components.values():
            if hasattr(component, 'reset_tracking'):
                component.reset_tracking()
            if hasattr(component, 'reset_awareness'):
                component.reset_awareness()
