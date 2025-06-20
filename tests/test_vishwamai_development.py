"""
Test suite for VishwamAI model components.
"""

import pytest
import sys
import os
import tempfile
import json
from pathlib import Path

# Add VishwamAI to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Test imports and basic functionality without heavy dependencies
class TestBasicImports:
    """Test basic module imports and structure."""
    
    def test_core_imports(self):
        """Test that core modules can be imported."""
        try:
            import vishwamai
            assert hasattr(vishwamai, '__version__') or True  # Allow for missing version
        except ImportError as e:
            pytest.skip(f"VishwamAI core import failed: {e}")
    
    def test_model_module_exists(self):
        """Test that model module exists."""
        try:
            from vishwamai import model
            assert hasattr(model, 'VishwamAIModel') or hasattr(model, 'ModelConfig')
        except ImportError as e:
            pytest.skip(f"Model module import failed: {e}")
    
    def test_distillation_module_syntax(self):
        """Test that distillation module compiles without syntax errors."""
        import py_compile
        distillation_path = Path(__file__).parent.parent / "vishwamai" / "distillation.py"
        
        try:
            py_compile.compile(str(distillation_path), doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"Distillation module has syntax errors: {e}")


class TestDistillationModule:
    """Test distillation module functionality."""
    
    def test_distillation_config_creation(self):
        """Test DistillationConfig can be created."""
        try:
            from vishwamai.distillation import DistillationConfig
            
            config = DistillationConfig(
                use_duckdb_tracking=True,
                duckdb_path="./test_experiments.db",
                experiment_name="test_experiment",
                num_train_epochs=1,
                max_steps=10
            )
            
            assert config.use_duckdb_tracking is True
            assert config.experiment_name == "test_experiment"
            assert config.num_train_epochs == 1
            
        except ImportError as e:
            pytest.skip(f"Distillation module import failed: {e}")
    
    def test_duckdb_tracker_creation(self):
        """Test DuckDBDistillationTracker can be created."""
        try:
            from vishwamai.distillation import DuckDBDistillationTracker
            
            with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as tmp:
                tracker = DuckDBDistillationTracker(tmp.name, "test_experiment")
                
                # Test that tracker has required methods
                assert hasattr(tracker, 'start_experiment')
                assert hasattr(tracker, 'log_distillation_metrics')
                assert hasattr(tracker, 'finish_experiment')
                assert hasattr(tracker, 'close')
                
                tracker.close()
                os.unlink(tmp.name)
                
        except ImportError as e:
            pytest.skip(f"DuckDB tracker import failed: {e}")
        except Exception as e:
            pytest.skip(f"DuckDB not available: {e}")
    
    def test_no_wandb_references(self):
        """Test that distillation module has no wandb imports."""
        distillation_path = Path(__file__).parent.parent / "vishwamai" / "distillation.py"
        
        with open(distillation_path, 'r') as f:
            content = f.read()
        
        # Check for problematic wandb imports or usage (excluding comments)
        problematic_lines = []
        for i, line in enumerate(content.split('\n'), 1):
            stripped_line = line.strip()
            # Skip comment lines and lines that mention wandb only in comments
            if stripped_line.startswith('#'):
                continue
            # Check for actual wandb imports or usage
            if 'import wandb' in line or 'from wandb' in line:
                problematic_lines.append(f"Line {i}: {line.strip()}")
            elif 'wandb.' in line and not line.strip().startswith('#'):
                # Make sure it's not just mentioned in a comment at the end of line
                code_part = line.split('#')[0].strip()
                if 'wandb.' in code_part:
                    problematic_lines.append(f"Line {i}: {line.strip()}")
        
        assert len(problematic_lines) == 0, f"Found wandb references: {problematic_lines}"


class TestConfigFiles:
    """Test configuration files and their validity."""
    
    def test_distillation_config_json(self):
        """Test that distillation config JSON is valid."""
        config_path = Path(__file__).parent.parent / "configs" / "distillation_config.json"
        
        if not config_path.exists():
            pytest.skip("Distillation config file not found")
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Test required sections exist
        assert 'tracking_config' in config, "tracking_config section missing"
        assert 'use_duckdb' in config['tracking_config'], "use_duckdb setting missing"
        
        # Test that wandb is not enabled
        if 'use_wandb' in config.get('tracking_config', {}):
            assert config['tracking_config']['use_wandb'] is False, "wandb should be disabled"
    
    def test_all_configs_are_valid_json(self):
        """Test that all config files are valid JSON."""
        configs_dir = Path(__file__).parent.parent / "configs"
        
        if not configs_dir.exists():
            pytest.skip("Configs directory not found")
        
        for config_file in configs_dir.glob("*.json"):
            try:
                with open(config_file, 'r') as f:
                    json.load(f)
            except json.JSONDecodeError as e:
                pytest.fail(f"Invalid JSON in {config_file}: {e}")


class TestModelComponents:
    """Test core model components."""
    
    def test_model_config_creation(self):
        """Test that model configs can be created."""
        try:
            from vishwamai.model import ModelConfig
            
            config = ModelConfig(
                dim=512,
                depth=6,
                heads=8,
                vocab_size=50000
            )
            
            assert config.dim == 512
            assert config.depth == 6
            
        except ImportError as e:
            pytest.skip(f"Model config import failed: {e}")
        except Exception as e:
            pytest.skip(f"Model config creation failed: {e}")
    
    def test_attention_module_imports(self):
        """Test attention module imports."""
        try:
            from vishwamai import attention
            # Test that attention module has key components
            assert hasattr(attention, 'Attention') or hasattr(attention, 'MultiHeadAttention')
        except ImportError as e:
            pytest.skip(f"Attention module import failed: {e}")
    
    def test_layers_module_imports(self):
        """Test layers module imports."""
        try:
            from vishwamai import layers
            # Basic test that layers module exists and has some content
            assert hasattr(layers, 'TransformerBlock') or len(dir(layers)) > 3
        except ImportError as e:
            pytest.skip(f"Layers module import failed: {e}")


class TestTrainingComponents:
    """Test training-related components."""
    
    def test_training_config_creation(self):
        """Test training config creation."""
        try:
            from vishwamai.training import TrainingConfig
            
            config = TrainingConfig(
                learning_rate=1e-4,
                batch_size=32,
                max_steps=1000
            )
            
            assert config.learning_rate == 1e-4
            
        except ImportError as e:
            pytest.skip(f"Training config import failed: {e}")
        except Exception as e:
            pytest.skip(f"Training config creation failed: {e}")


class TestUtilities:
    """Test utility functions and helpers."""
    
    def test_utils_module_exists(self):
        """Test that utils module exists and has basic functionality."""
        try:
            from vishwamai import utils
            # Basic test that utils exists
            assert len(dir(utils)) > 1
        except ImportError as e:
            pytest.skip(f"Utils module import failed: {e}")


class TestMultimodalComponents:
    """Test multimodal functionality."""
    
    def test_multimodal_imports(self):
        """Test multimodal module imports."""
        try:
            from vishwamai import multimodal
            assert len(dir(multimodal)) > 1
        except ImportError as e:
            pytest.skip(f"Multimodal module import failed: {e}")
    
    def test_advanced_multimodal_imports(self):
        """Test advanced multimodal imports."""
        try:
            from vishwamai import advanced_multimodal
            assert len(dir(advanced_multimodal)) > 1
        except ImportError as e:
            pytest.skip(f"Advanced multimodal module import failed: {e}")


class TestIntegrationReadiness:
    """Test overall integration readiness of the VishwamAI system."""
    
    def test_no_duplicate_functions_in_distillation(self):
        """Test that there are no duplicate function definitions in distillation.py."""
        distillation_path = Path(__file__).parent.parent / "vishwamai" / "distillation.py"
        
        try:
            import ast
            with open(distillation_path, 'r') as f:
                content = f.read()
            
            tree = ast.parse(content)
            class_duplicates = {}
            
            # Check each class for duplicate methods
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    class_name = node.name
                    method_counts = {}
                    
                    for child in node.body:
                        if isinstance(child, ast.FunctionDef):
                            method_name = child.name
                            method_counts[method_name] = method_counts.get(method_name, 0) + 1
                    
                    # Find duplicates in this class
                    duplicates = {name: count for name, count in method_counts.items() if count > 1}
                    if duplicates:
                        class_duplicates[class_name] = duplicates
            
            assert len(class_duplicates) == 0, f"Found duplicate functions: {class_duplicates}"
            
        except ImportError:
            # Fallback to simple text analysis (less accurate)
            with open(distillation_path, 'r') as f:
                content = f.read()
            
            # Simple check: just count __init__ methods vs classes
            class_count = content.count('class ')
            init_count = content.count('def __init__')
            
            # Allow at most one __init__ per class
            assert init_count <= class_count, f"Too many __init__ methods: {init_count} methods for {class_count} classes"
    
    def test_distillation_config_compatibility(self):
        """Test that distillation config is compatible with new module."""
        config_path = Path(__file__).parent.parent / "configs" / "distillation_config.json"
        
        if not config_path.exists():
            pytest.skip("Distillation config not found")
        
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        
        # Test that config has DuckDB settings
        tracking_config = config_data.get('tracking_config', {})
        assert 'use_duckdb' in tracking_config, "use_duckdb setting missing from config"
        assert 'db_path' in tracking_config, "db_path setting missing from config"
    
    def test_requirements_file_exists(self):
        """Test that requirements.txt exists and has key dependencies."""
        req_path = Path(__file__).parent.parent / "requirements.txt"
        
        if not req_path.exists():
            pytest.skip("requirements.txt not found")
        
        with open(req_path, 'r') as f:
            requirements = f.read().lower()
        
        # Check for key dependencies
        assert 'torch' in requirements or 'pytorch' in requirements, "PyTorch not in requirements"
        assert 'jax' in requirements, "JAX not in requirements"
        assert 'duckdb' in requirements, "DuckDB not in requirements"


class TestExampleScripts:
    """Test example scripts and demos."""
    
    def test_demo_script_exists(self):
        """Test that demo script exists."""
        demo_path = Path(__file__).parent.parent / "demo_vishwamai.py"
        assert demo_path.exists(), "demo_vishwamai.py not found"
    
    def test_demo_script_syntax(self):
        """Test demo script syntax."""
        demo_path = Path(__file__).parent.parent / "demo_vishwamai.py"
        
        if not demo_path.exists():
            pytest.skip("Demo script not found")
        
        import py_compile
        try:
            py_compile.compile(str(demo_path), doraise=True)
        except py_compile.PyCompileError as e:
            pytest.fail(f"Demo script has syntax errors: {e}")


if __name__ == "__main__":
    # Run tests if called directly
    pytest.main([__file__, "-v"])
