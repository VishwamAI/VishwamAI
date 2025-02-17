"""
Vishwamai - Advanced AI Training Framework
"""

# Core components with fixed import order to avoid circular dependencies 
from .config import ModelArgs
from .constants import WORLD_SIZE, ATTN_IMPL, BLOCK_SIZE
from .base_layers import Linear
from .utils import precompute_freqs_cis
from .tokenizer import VishwamAITokenizer, TokenizerConfig
from .parallel import ColumnParallelLinear, RMSNorm, ParallelEmbedding
from .Transformer import Transformer
# Move model_factory import after all core components
from .model_factory import create_model

__version__ = "0.1.1"
__author__ = "Vishwamai Contributors"

__all__ = [
    'ModelArgs',
    'WORLD_SIZE', 
    'ATTN_IMPL',
    'BLOCK_SIZE',
    'Linear',
    'precompute_freqs_cis',
    'VishwamAITokenizer',
    'TokenizerConfig', 
    'ColumnParallelLinear',
    'RMSNorm',
    'ParallelEmbedding',
    'Transformer',
    'create_model'
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
