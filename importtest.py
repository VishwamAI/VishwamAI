"""Test script to verify all imports are working properly."""

import os
import sys
from unittest.mock import patch

os.environ["JAX_PLATFORMS"] = "cpu"  # Force CPU to avoid TPU/GPU initialization issues

def mock_create_device_mesh(*args, **kwargs):
    """Mock device mesh creation to return a simple 1-device mesh."""
    import jax.numpy as jnp
    return jnp.array([[0]])

def test_imports():
    # Set up minimal device mesh for testing
    import jax
    jax.config.update('jax_platform_name', 'cpu')
    
    # Patch mesh creation before importing CUDA modules
    import jax._src.mesh_utils as mesh_utils
    mesh_utils.create_device_mesh = mock_create_device_mesh
    
    try:
        print("Testing JAX imports...")
        import jax
        import jax.numpy as jnp
        print("✓ JAX imports successful")
        
        print("\nTesting core VishwamAI imports...")
        from vishwamai.model import VishwamAI, VishwamAIConfig
        from vishwamai.transformer import EnhancedTransformerModel, TPUTrainingState
        from vishwamai.pipeline import TPUDataPipeline, DistillationDataPipeline
        from vishwamai.device_mesh import TPUMeshContext
        print("✓ Core VishwamAI imports successful")
        
        print("\nTesting distillation imports...")
        from vishwamai.distill import (
            create_student_model,
            initialize_from_teacher,
            DistillationTrainer
        )
        print("✓ Distillation imports successful")
        
        print("\nTesting thoughts imports...")
        from vishwamai.thoughts import TreeOfThoughts
        print("✓ Thoughts imports successful")
        
        print("\nTesting config imports...")
        from vishwamai.configs.tpu_v3_config import TPUV3Config
        from vishwamai.configs.budget_model_config import BudgetModelConfig
        print("✓ Config imports successful")
        
        print("\nTesting utility imports...")
        from vishwamai.profiler import TPUProfiler
        from tqdm.auto import tqdm
        print("✓ Utility imports successful")
        
        print("\nAll imports successful!")
        return True
        
    except ImportError as e:
        print(f"\n❌ Import failed: {str(e)}")
        return False

if __name__ == "__main__":
    test_imports()
