"""
VishwamAI Extensions Package

This package contains specialized modules and advanced features including text generation,
emergent behavior analysis, ethical frameworks, and memory augmentation.
"""

from .generate import generate_text, generate_with_beam_search
from .emergent_behavior import analyze_behavior, detect_patterns
from .ethical_framework import EthicalChecker, validate_output
from .neural_memory import MemoryBank, store_memory, retrieve_memory
from .integrated_information import compute_phi, analyze_integration
from .tree_of_thoughts import TreeOfThoughts, explore_thoughts
from .open_ended_learning import adaptive_learning, explore_concepts

__all__ = [
    # Text generation
    'generate_text',
    'generate_with_beam_search',
    
    # Behavior analysis
    'analyze_behavior',
    'detect_patterns',
    
    # Ethical framework
    'EthicalChecker',
    'validate_output',
    
    # Memory systems
    'MemoryBank',
    'store_memory',
    'retrieve_memory',
    
    # Information integration
    'compute_phi',
    'analyze_integration',
    
    # Advanced reasoning
    'TreeOfThoughts',
    'explore_thoughts',
    
    # Continuous learning
    'adaptive_learning',
    'explore_concepts',
]
