# Import required modules
import sys
import platform
import importlib.util
import warnings

# Dictionary to store import status
import_status = {}

def safe_import(module_name):
    """Safely import a module and return None if import fails"""
    try:
        if module_name in sys.modules:
            return sys.modules[module_name]
        return importlib.import_module(module_name)
    except ImportError as e:
        warnings.warn(f"Failed to import {module_name}: {str(e)}")
        return None

# Basic imports
import_status['torch'] = safe_import('torch')
import_status['matplotlib.pyplot'] = safe_import('matplotlib.pyplot')
import_status['pandas'] = safe_import('pandas')

# Optional dependencies
import_status['transformers'] = safe_import('transformers')
import_status['datasets'] = safe_import('datasets')

# Import required modules conditionally
torch = import_status['torch']
plt = import_status['matplotlib.pyplot']
pd = import_status['pandas']

# VishwamAI components
try:
    from vishwamai.base_layers import Linear
    from vishwamai.Transformer import Transformer
    from vishwamai import (
        create_model,
        ModelArgs,
        VishwamAITokenizer,
        TokenizerConfig
    )
    from vishwamai.advanced_training import AdvancedTrainer
    from vishwamai.fp8_cast_bf16 import main as fp8_main
    from vishwamai.neural_memory import NeuralMemory
    from vishwamai.tree_of_thoughts import TreeConfig, RewardConfig
    from vishwamai.curriculum import CurriculumConfig
    import_status['vishwamai'] = True
except ImportError as e:
    import_status['vishwamai'] = False
    print(f"Error importing VishwamAI components: {str(e)}")

def check_system_compatibility():
    """Check if the system meets all requirements"""
    print("\nSystem Information:")
    print(f"Python version: {sys.version}")
    print(f"Operating System: {platform.system()} {platform.release()}")
    
    compatibility_status = True
    
    # Check PyTorch
    if torch:
        print(f"PyTorch version: {torch.__version__}")
        if hasattr(torch, 'cuda') and torch.cuda.is_available():
            print(f"CUDA available: Yes (Device: {torch.cuda.get_device_name(0)})")
        else:
            print("CUDA available: No (Running on CPU)")
    else:
        print("PyTorch: Not installed")
        compatibility_status = False
    
    # Check other dependencies
    print("\nDependency Status:")
    for module, status in import_status.items():
        print(f"{module}: {'✓ Installed' if status else '✗ Not installed'}")
        if not status and module in ['torch', 'vishwamai']:
            compatibility_status = False
    
    return compatibility_status

def test_imports():
    """Test imports and basic functionality"""
    print("\nTesting imports and system compatibility...")
    
    if not check_system_compatibility():
        print("\n⚠️ System compatibility check failed. Some features may not work correctly.")
        return False
    
    if not import_status.get('vishwamai'):
        print("\n❌ Critical VishwamAI components failed to import. Cannot proceed with tests.")
        return False
    
    success = True
    try:
        print("\nTesting VishwamAI components:")
        
        # Test ModelArgs
        args = ModelArgs(
            max_batch_size=4,
            max_seq_len=2048,
            dtype="fp8",
            vocab_size=32000,
            dim=1024,
            n_layers=12,
            n_heads=16
        )
        print("✓ ModelArgs")
        
        # Only test components if previous tests passed
        if args:
            try:
                transformer = Transformer(args)
                print("✓ Transformer")
            except Exception as e:
                print(f"✗ Transformer: {str(e)}")
                success = False
            
            try:
                tokenizer = VishwamAITokenizer(TokenizerConfig(
                    vocab_size=32000,
                    max_sentence_length=2048
                ))
                print("✓ VishwamAITokenizer")
            except Exception as e:
                print(f"✗ VishwamAITokenizer: {str(e)}")
                success = False
    
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        success = False
    
    if success:
        print("\n✅ All available components tested successfully")
    else:
        print("\n⚠️ Some tests failed - check logs above for details")
    
    return success

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
