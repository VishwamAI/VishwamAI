#!/usr/bin/env python3
"""
Setup and validation script for VishwamAI.

This script checks dependencies, validates configurations,
and provides setup guidance.
"""

import sys
import importlib
import subprocess
from pathlib import Path
import json

def check_python_version():
    """Check if Python version is compatible."""
    if sys.version_info < (3, 8):
        print("❌ Python 3.8+ required")
        return False
    print(f"✅ Python {sys.version_info.major}.{sys.version_info.minor}")
    return True

def check_dependencies():
    """Check if required dependencies are installed."""
    
    required_packages = [
        ('jax', 'JAX'),
        ('jaxlib', 'JAXLib'),
        ('flax', 'Flax'),
        ('optax', 'Optax'),
        ('numpy', 'NumPy'),
        ('chex', 'Chex'),
        ('einops', 'Einops'),
    ]
    
    optional_packages = [
        ('transformers', 'Transformers (for tokenizers)'),
        ('datasets', 'Datasets (for data loading)'),
        ('wandb', 'Weights & Biases (for logging)'),
        ('torch', 'PyTorch (for some utilities)'),
        ('PIL', 'Pillow (for image processing)'),
        ('librosa', 'Librosa (for audio processing)'),
    ]
    
    print("🔍 Checking required dependencies...")
    missing_required = []
    
    for package, name in required_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"❌ {name}")
            missing_required.append(package)
    
    print("\n🔍 Checking optional dependencies...")
    missing_optional = []
    
    for package, name in optional_packages:
        try:
            importlib.import_module(package)
            print(f"✅ {name}")
        except ImportError:
            print(f"⚠️  {name} (optional)")
            missing_optional.append(package)
    
    return missing_required, missing_optional

def install_dependencies(packages):
    """Install missing dependencies."""
    if not packages:
        return
    
    print(f"\n📦 Installing missing packages: {', '.join(packages)}")
    
    try:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install'
        ] + packages)
        print("✅ Installation completed")
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")

def check_hardware():
    """Check available hardware."""
    print("\n🖥️  Checking hardware...")
    
    try:
        import jax
        devices = jax.devices()
        
        print(f"Available devices: {len(devices)}")
        for i, device in enumerate(devices):
            print(f"  {i}: {device}")
        
        # Test basic JAX operations
        import jax.numpy as jnp
        x = jnp.array([1, 2, 3])
        y = jnp.sum(x)
        print(f"✅ JAX operations working (test sum: {y})")
        
        return True
        
    except Exception as e:
        print(f"❌ Hardware check failed: {e}")
        return False

def validate_configs():
    """Validate configuration files."""
    print("\n📋 Validating configuration files...")
    
    config_dir = Path(__file__).parent / "configs"
    
    if not config_dir.exists():
        print(f"❌ Config directory not found: {config_dir}")
        return False
    
    config_files = list(config_dir.glob("*.json"))
    
    if not config_files:
        print("❌ No configuration files found")
        return False
    
    valid_configs = []
    
    for config_file in config_files:
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
            
            # Basic validation
            if 'model_config' not in config:
                print(f"⚠️  {config_file.name}: missing model_config")
                continue
            
            model_config = config['model_config']
            required_fields = ['dim', 'depth', 'heads', 'vocab_size']
            
            missing_fields = [field for field in required_fields if field not in model_config]
            if missing_fields:
                print(f"⚠️  {config_file.name}: missing fields {missing_fields}")
                continue
            
            print(f"✅ {config_file.name}")
            valid_configs.append(config_file)
            
        except json.JSONDecodeError as e:
            print(f"❌ {config_file.name}: invalid JSON - {e}")
        except Exception as e:
            print(f"❌ {config_file.name}: validation error - {e}")
    
    return len(valid_configs) > 0

def create_example_script():
    """Create a simple example script."""
    
    example_script = '''#!/usr/bin/env python3
"""
Simple VishwamAI example script.
"""

import jax
import jax.numpy as jnp
from vishwamai import ModelConfig, VishwamAIModel, get_hardware_info

def main():
    print("🤖 VishwamAI Example")
    
    # Check hardware
    hw_info = get_hardware_info()
    print(f"Hardware: {hw_info['num_devices']} devices")
    
    # Create small model configuration
    config = ModelConfig(
        dim=512,
        depth=6,
        heads=8,
        vocab_size=1000,
        max_seq_len=128
    )
    
    # Create model
    model = VishwamAIModel(config)
    
    # Initialize with dummy input
    rng = jax.random.PRNGKey(42)
    dummy_input = jnp.ones((1, 10), dtype=jnp.int32)
    params = model.init(rng, dummy_input, training=False)
    
    # Forward pass
    output = model.apply(params, dummy_input, training=False)
    print(f"Model output shape: {output.shape}")
    print("✅ Example completed successfully!")

if __name__ == '__main__':
    main()
'''
    
    example_path = Path(__file__).parent / "example.py"
    
    with open(example_path, 'w') as f:
        f.write(example_script)
    
    print(f"📄 Created example script: {example_path}")
    return example_path

def main():
    print("🚀 VishwamAI Setup and Validation")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Check dependencies
    missing_required, missing_optional = check_dependencies()
    
    if missing_required:
        print(f"\n❌ Missing required dependencies: {missing_required}")
        install_choice = input("Install missing required dependencies? (y/n): ").lower()
        
        if install_choice == 'y':
            install_dependencies(missing_required)
        else:
            print("❌ Cannot proceed without required dependencies")
            sys.exit(1)
    
    if missing_optional:
        print(f"\n⚠️  Missing optional dependencies: {missing_optional}")
        install_choice = input("Install missing optional dependencies? (y/n): ").lower()
        
        if install_choice == 'y':
            install_dependencies(missing_optional)
    
    # Check hardware
    if not check_hardware():
        print("⚠️  Hardware check failed - some features may not work")
    
    # Validate configurations
    if not validate_configs():
        print("⚠️  Configuration validation failed")
    
    # Create example script
    example_path = create_example_script()
    
    print("\n🎉 Setup validation completed!")
    print("\nNext steps:")
    print(f"1. Run the example: python {example_path}")
    print("2. Train a model: python scripts/train_vishwamai.py --config configs/small_model.json")
    print("3. Run inference: python scripts/inference.py --prompt 'Hello, world!'")
    print("\nFor more information, see the documentation in docs/")

if __name__ == '__main__':
    main()
