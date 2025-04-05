import os
import sys
from unittest.mock import patch

# Force CPU to avoid TPU/GPU initialization issues
os.environ["JAX_PLATFORMS"] = "cpu"

def mock_create_device_mesh(*args, **kwargs):
    """Mock device mesh creation to return a simple 1-device mesh."""
    import jax.numpy as jnp
    return jnp.array([[0]])

def test_imports():
    """Function to test all necessary imports for the VishwamAI framework."""
    try:
        print("Testing JAX imports...")
        import jax
        import jax.numpy as jnp
        jax.config.update('jax_platform_name', 'cpu')
        print("✓ JAX imports successful")

        print("\nPatching JAX mesh utilities...")
        try:
            import jax._src.mesh_utils as mesh_utils
            mesh_utils.create_device_mesh = mock_create_device_mesh
            print("✓ JAX mesh utilities patched successfully")
        except ImportError:
            print("⚠️ Could not patch JAX mesh utilities (optional)")

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

        print("\n🎉 All imports successful!")
        return True

    except ImportError as e:
        print(f"\n❌ Import failed: {e.__class__.__name__}: {str(e)}")
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {e.__class__.__name__}: {str(e)}")
        return False

__all__ = [
    'VishwamAI',
    'VishwamAIConfig',
    'EnhancedTransformerModel',
    'TPUTrainingState',
    'TPUDataPipeline',
    'DistillationDataPipeline',
    'TPUMeshContext',
    'create_student_model',
    'initialize_from_teacher',
    'DistillationTrainer',
    'TreeOfThoughts',
    'TPUV3Config',
    'BudgetModelConfig',
    'TPUProfiler',
    'tqdm',
]  # type: ignore[assignment]

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
