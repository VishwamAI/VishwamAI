import os
import sys
import unittest
from unittest.mock import patch

# Force CPU to avoid TPU/GPU initialization issues
os.environ["JAX_PLATFORMS"] = "cpu"

def mock_create_device_mesh(*args, **kwargs):
    """Mock device mesh creation to return a simple 1-device mesh."""
    import jax.numpy as jnp
    return jnp.array([[0]])

class TestConfigImports(unittest.TestCase):
    """Test case for config imports."""

    def test_config_imports(self):
        """Test all necessary config imports for the VishwamAI framework."""
        try:
            print("Testing JAX imports...")
            import jax
            import jax.numpy as jnp
            jax.config.update('jax_platform_name', 'cpu')
            print("✓ JAX imports successful")

            print("\nPatching JAX mesh utilities...")
            import jax._src.mesh_utils as mesh_utils
            mesh_utils.create_device_mesh = mock_create_device_mesh
            print("✓ JAX mesh utilities patched successfully")

            print("\nTesting config imports...")
            from vishwamai.configs.tpu_v3_config import TPUV3Config
            from vishwamai.configs.budget_model_config import BudgetModelConfig
            print("✓ Config imports successful")
            
            print("\nAll config imports successful!")
            self.assertTrue(True)

        except ImportError as e:
            print(f"\n❌ Import failed: {str(e)}")
            self.fail(f"Import failed: {str(e)}")

if __name__ == "__main__":
    unittest.main()
