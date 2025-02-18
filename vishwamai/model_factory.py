import torch
import torch.nn as nn
from typing import Dict, Optional, Union
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

def create_model(config: Union[dict, ModelArgs], device: Optional[torch.device] = None) -> nn.Module:
    """
    Create a model instance based on configuration.
    Args:
        config: Either a dictionary with model configuration or a ModelArgs instance
    """
    # Convert dict keys to match ModelArgs attributes
    def _convert_dict_to_model_args(config_dict):
        # Map dict keys to ModelArgs attributes
        key_mapping = {
            'hidden_size': 'dim',
            'dim': 'dim',  # Allow both hidden_size and dim
            'num_hidden_layers': 'n_layers',
            'num_attention_heads': 'n_heads',
            'intermediate_size': 'inter_dim'
        }
        
        # Convert input dict to match ModelArgs attribute names
        converted = {}
        for k, v in config_dict.items():
            if k in key_mapping:
                converted[key_mapping[k]] = v
            else:
                converted[k] = v
        return converted

    # Handle input types
    if isinstance(config, ModelArgs):
        model_args = config
        is_moe = model_args.n_routed_experts > 0
    else:
        # Convert dictionary keys to match ModelArgs attributes
        config_dict = _convert_dict_to_model_args(config)
        is_moe = config_dict.get("model_type") == "moe"
        
        # Set default values for missing keys
        config_dict.setdefault("vocab_size", 102400)
        config_dict.setdefault("max_seq_len", 32768)
        config_dict.setdefault("inter_dim", config_dict.get("dim", 2048) * 4)  # Common ratio for transformer
        
        model_args = ModelArgs(
            dim=config_dict.get("dim", config_dict.get("hidden_size", 2048)),
            n_layers=config_dict["n_layers"],
            n_heads=config_dict["n_heads"],
            vocab_size=config_dict.get("vocab_size", 102400),
            n_dense_layers=0 if is_moe else config_dict["n_layers"],
            inter_dim=config_dict["inter_dim"],
            max_seq_len=config_dict.get("max_seq_len", config_dict.get("max_position_embeddings", 32768)),  # Extended for 7B
            
            # MoE specific parameters
            n_routed_experts=config_dict.get("num_experts", 128) if is_moe else 0,
            n_shared_experts=config_dict.get("num_shared_experts", 4) if is_moe else 0,
            n_activated_experts=config_dict.get("num_activated_experts", 8) if is_moe else 0,
            n_expert_groups=config_dict.get("num_expert_groups", 2),
            moe_inter_dim=config_dict.get("moe_intermediate_size", 2048),
            
            # Advanced features
            dtype=torch.bfloat16 if config_dict.get("mixed_precision") != "fp8" else torch.float8_e4m3fn,
            gradient_checkpointing=config_dict.get("gradient_checkpointing", True),
            use_alibi=config_dict.get("use_alibi", True),
            use_rope_scaling=config_dict.get("use_rope_scaling", True),
            parallel_attn=True,
            
            # Attention dimensions
            qk_nope_head_dim=config_dict.get("qk_nope_head_dim", 256),
            qk_rope_head_dim=config_dict.get("qk_rope_head_dim", 128),
            v_head_dim=config_dict.get("v_head_dim", 256),
            
            # Scaling parameters
            rope_theta=config_dict.get("rope_theta", 20000.0),
            rope_factor=config_dict.get("rope_factor", 80),
            mscale=config_dict.get("mscale", 1.5),
            rope_condense_ratio=config_dict.get("rope_condense_ratio", 1.2)
        )

    # Create model with device
    model = Transformer(model_args, device=device)
    if is_moe:
        model = MoE(model)
    
    # Move model to device if specified
    if device is not None:
        model = model.to(device)
    
    # Create tokenizer from ModelArgs
    tokenizer = VishwamAITokenizer(TokenizerConfig(
        vocab_size=model_args.vocab_size,
        max_sentence_length=model_args.max_seq_len
    ))
    
    # Create advanced config from ModelArgs
    advanced_config = AdvancedModelConfig(
        hidden_dim=model_args.dim,
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
