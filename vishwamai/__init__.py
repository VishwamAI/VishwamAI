"""
Vishwamai - Advanced AI Training Framework
=======================================

A sophisticated training framework implementing advanced AI concepts:
- Emergent behavior
- Integrated information processing
- Ethical compliance
- Hardware optimization
- Curriculum learning
- Open-ended learning
"""

from .trainer_unified import UnifiedTrainer, UnifiedTrainerConfig
from .model_factory import ModelFactory, AdvancedModelConfig
from .curriculum import CurriculumConfig, CurriculumScheduler
from .emergent_behavior import EmergentConfig, EmergentBehaviorModule
from .integrated_information import IntegrationConfig, IntegratedInformationModule
from .ethical_framework import EthicalConfig, EthicalFramework
from .hardware_adapter import HardwareConfig, HardwareAdapter
from .open_ended_learning import OpenEndedConfig, OpenEndedLearning

__version__ = "0.1.1"
__author__ = "Vishwamai Contributors"

__all__ = [
    'UnifiedTrainer',
    'UnifiedTrainerConfig',
    'ModelFactory',
    'AdvancedModelConfig',
    'CurriculumConfig',
    'CurriculumScheduler',
    'EmergentConfig',
    'EmergentBehaviorModule',
    'IntegrationConfig',
    'IntegratedInformationModule',
    'EthicalConfig',
    'EthicalFramework',
    'HardwareConfig',
    'HardwareAdapter',
    'OpenEndedConfig',
    'OpenEndedLearning',
]

# Module level metadata
_metadata = {
    'framework_name': 'Vishwamai',
    'description': 'Advanced AI Training Framework',
    'license': 'MIT',
    'requires_python': '>=3.8',
    'maintainers': ['Vishwamai Contributors'],
    'repository': 'https://github.com/yourusername/vishwamai',
    'documentation': 'https://vishwamai.readthedocs.io',
    'components': {
        'curriculum': 'Adaptive curriculum learning and task progression',
        'emergent': 'Self-organizing behavior and novelty detection',
        'integration': 'Consciousness-inspired information processing',
        'ethical': 'Ethical decision making and monitoring',
        'hardware': 'Hardware-specific optimizations and adaptations',
        'open_ended': 'Continuous evolution and task generation'
    }
}

def get_framework_info() -> dict:
    """Get information about the Vishwamai framework."""
    return _metadata.copy()

def print_framework_info():
    """Print formatted information about the Vishwamai framework."""
    info = get_framework_info()
    print(f"\n{info['framework_name']} v{__version__}")
    print("=" * (len(info['framework_name']) + len(__version__) + 3))
    print(f"\n{info['description']}")
    print("\nComponents:")
    for component, desc in info['components'].items():
        print(f"- {component}: {desc}")
    print(f"\nPython required: {info['requires_python']}")
    print(f"License: {info['license']}")
    print(f"\nDocumentation: {info['documentation']}")
    print(f"Repository: {info['repository']}")
