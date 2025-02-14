"""
VishwamAI Examples Package
Provides example implementations and usage patterns for the VishwamAI model.
"""

from .model_usage import EnhancedVishwamAI
from .model_examples import (
    mathematical_reasoning_example,
    scientific_analysis_example,
    code_generation_example,
    multi_step_reasoning_example,
    research_synthesis_example,
    batch_processing_example
)
from .pretrain_and_upload import ModelPretrainer, PretrainConfig

__all__ = [
    # Main interface
    'EnhancedVishwamAI',
    
    # Example functions
    'mathematical_reasoning_example',
    'scientific_analysis_example',
    'code_generation_example',
    'multi_step_reasoning_example',
    'research_synthesis_example',
    'batch_processing_example',
    
    # Training components
    'ModelPretrainer',
    'PretrainConfig'
]

# Example usage documentation
USAGE_EXAMPLES = {
    'mathematical_reasoning': """
    # Mathematical reasoning with tree of thoughts
    from vishwamai.examples import mathematical_reasoning_example
    
    mathematical_reasoning_example()
    """,
    
    'scientific_analysis': """
    # Scientific paper analysis with neural memory
    from vishwamai.examples import scientific_analysis_example
    
    scientific_analysis_example()
    """,
    
    'code_generation': """
    # Code generation with cache augmentation
    from vishwamai.examples import code_generation_example
    
    code_generation_example()
    """,
    
    'multi_step_reasoning': """
    # Complex reasoning with all components
    from vishwamai.examples import multi_step_reasoning_example
    
    multi_step_reasoning_example()
    """,
    
    'research_synthesis': """
    # Research synthesis across papers
    from vishwamai.examples import research_synthesis_example
    
    research_synthesis_example()
    """,
    
    'batch_processing': """
    # Efficient batch processing
    from vishwamai.examples import batch_processing_example
    
    batch_processing_example()
    """,
    
    'model_training': """
    # Full model pretraining
    from vishwamai.examples import ModelPretrainer, PretrainConfig
    
    config = PretrainConfig(
        output_dir="pretrain_output",
        num_epochs=3,
        batch_size=8
    )
    
    trainer = ModelPretrainer(config)
    trainer.initialize_components()
    trainer.train()
    trainer.upload_to_hub()
    """
}

def print_examples():
    """Print all available examples with usage instructions."""
    print("VishwamAI Examples:")
    print("-" * 50)
    
    for name, usage in USAGE_EXAMPLES.items():
        print(f"\n{name.replace('_', ' ').title()}:")
        print(usage)
        print("-" * 50)

def run_all_examples():
    """Run all example functions."""
    examples = [
        mathematical_reasoning_example,
        scientific_analysis_example,
        code_generation_example,
        multi_step_reasoning_example,
        research_synthesis_example,
        batch_processing_example
    ]
    
    for example in examples:
        try:
            print(f"\nRunning {example.__name__}...")
            example()
        except Exception as e:
            print(f"Error in {example.__name__}: {str(e)}")
