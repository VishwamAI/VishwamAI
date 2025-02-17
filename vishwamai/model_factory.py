import torch
import torch.nn as nn
from typing import Dict, Optional
from dataclasses import dataclass
from .emergent_behavior import EmergentBehaviorModule, EmergentConfig
from .integrated_information import IntegratedInformationModule, IntegrationConfig
from .ethical_framework import EthicalFramework, EthicalConfig
from .hardware_adapter import HardwareAdapter, HardwareConfig
from .open_ended_learning import OpenEndedLearning, OpenEndedConfig

@dataclass
class AdvancedModelConfig:
    """Configuration for advanced model components."""
    hidden_dim: int = 768
    emergent_config: Optional[EmergentConfig] = None
    integration_config: Optional[IntegrationConfig] = None
    ethical_config: Optional[EthicalConfig] = None
    hardware_config: Optional[HardwareConfig] = None
    open_ended_config: Optional[OpenEndedConfig] = None

class ModelFactory:
    """Factory for creating and initializing advanced model components."""
    
    @staticmethod
    def create_advanced_components(
        base_model: nn.Module,
        config: AdvancedModelConfig
    ) -> Dict[str, nn.Module]:
        """
        Create all advanced components for the model.
        
        Args:
            base_model: Base model to enhance
            config: Configuration for all components
            
        Returns:
            Dictionary of initialized components
        """
        components = {}
        
        # Create emergent behavior module
        components['emergent'] = EmergentBehaviorModule(
            hidden_dim=config.hidden_dim,
            config=config.emergent_config
        )
        
        # Create integrated information module
        components['integration'] = IntegratedInformationModule(
            hidden_dim=config.hidden_dim,
            config=config.integration_config
        )
        
        # Create ethical framework
        components['ethical'] = EthicalFramework(
            hidden_dim=config.hidden_dim,
            config=config.ethical_config
        )
        
        # Create hardware adapter
        components['hardware'] = HardwareAdapter(
            model=base_model,
            config=config.hardware_config
        )
        
        # Create open-ended learning module
        components['open_ended'] = OpenEndedLearning(
            hidden_dim=config.hidden_dim,
            config=config.open_ended_config
        )
        
        return components
    
    @staticmethod
    def initialize_components(components: Dict[str, nn.Module], device: torch.device):
        """Initialize components and move to device."""
        for name, component in components.items():
            component.to(device)
            if hasattr(component, 'reset_tracking'):
                component.reset_tracking()
            if hasattr(component, 'optimize_for_hardware'):
                component.optimize_for_hardware(
                    torch.randn(1, components['open_ended'].hidden_dim).to(device)
                )
    
    @staticmethod
    def verify_compatibility(components: Dict[str, nn.Module]) -> bool:
        """Verify all components are compatible with each other."""
        hidden_dims = set()
        for name, component in components.items():
            if hasattr(component, 'hidden_dim'):
                hidden_dims.add(component.hidden_dim)
        
        return len(hidden_dims) == 1
    
    @staticmethod
    def get_component_stats(components: Dict[str, nn.Module]) -> Dict[str, Dict]:
        """Get statistics from all components."""
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
    def save_components(
        components: Dict[str, nn.Module],
        path: str
    ):
        """Save all component states."""
        component_states = {
            name: component.state_dict()
            for name, component in components.items()
        }
        torch.save(component_states, path)
    
    @staticmethod
    def load_components(
        components: Dict[str, nn.Module],
        path: str
    ):
        """Load component states."""
        component_states = torch.load(path)
        for name, state in component_states.items():
            if name in components:
                components[name].load_state_dict(state)
                
    @staticmethod
    def reset_components(components: Dict[str, nn.Module]):
        """Reset all components to initial state."""
        for component in components.values():
            if hasattr(component, 'reset_tracking'):
                component.reset_tracking()
            if hasattr(component, 'reset_awareness'):
                component.reset_awareness()
