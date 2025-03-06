"""Validation script for QwQ-32B distillation setup."""
import os
import jax
import jax.numpy as jnp
import safetensors.flax as stf
from huggingface_hub import snapshot_download
import sys

def validate_environment():
    """Validate JAX and TPU setup."""
    print("Checking JAX and TPU setup...")
    try:
        devices = jax.devices()
        print(f"Found {len(devices)} devices: {devices}")
        if not any('TPU' in str(d) for d in devices):
            print("WARNING: No TPU devices found!")
            return False
    except Exception as e:
        print(f"Error checking devices: {str(e)}")
        return False
    return True

def validate_model_files():
    """Validate QwQ-32B model files."""
    print("\nChecking QwQ-32B model files...")
    try:
        model_path = snapshot_download(
            "Qwen/QwQ-32B",
            allow_patterns=["*.safetensors", "config.json", "tokenizer.model"],
            resume_download=True
        )
        
        # Check safetensor shards
        shard_files = [f for f in os.listdir(model_path) if f.endswith('.safetensors')]
        print(f"Found {len(shard_files)} safetensor shards")
        
        if len(shard_files) != 14:
            print(f"ERROR: Expected 14 safetensor files, found {len(shard_files)}")
            return False
        
        # Verify shard integrity
        for shard in sorted(shard_files):
            print(f"Validating {shard}...", end='')
            try:
                shard_path = os.path.join(model_path, shard)
                _ = stf.load_file(shard_path)
                print("OK")
            except Exception as e:
                print(f"FAILED: {str(e)}")
                return False
                
    except Exception as e:
        print(f"Error validating model files: {str(e)}")
        return False
    return True

def validate_project_structure():
    """Validate project directory structure."""
    print("\nChecking project structure...")
    required_dirs = ['checkpoints', 'logs', 'data']
    required_files = ['requirements.txt', 'train_vishwamai_distillation.ipynb', 'configs/distillation_config.yaml']
    
    missing = []
    for d in required_dirs:
        if not os.path.isdir(d):
            missing.append(d)
    
    for f in required_files:
        if not os.path.isfile(f):
            missing.append(f)
    
    if missing:
        print(f"Missing required files/directories: {missing}")
        return False
    
    print("All required files and directories present")
    return True

def validate_memory():
    """Validate system memory."""
    print("\nChecking system memory...")
    try:
        import psutil
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024 ** 3)
        available_gb = memory.available / (1024 ** 3)
        
        print(f"Total Memory: {total_gb:.1f}GB")
        print(f"Available Memory: {available_gb:.1f}GB")
        
        if total_gb < 120:  # Recommend at least 128GB
            print("WARNING: Less than recommended memory (128GB)")
            return False
            
    except ImportError:
        print("Could not check memory - psutil not installed")
        return False
    return True

def main():
    """Run all validation checks."""
    print("Running VishwamAI QwQ-32B Setup Validation")
    print("=" * 50)
    
    checks = [
        (validate_environment, "Environment"),
        (validate_model_files, "Model Files"),
        (validate_project_structure, "Project Structure"),
        (validate_memory, "System Memory")
    ]
    
    failed = []
    for check_fn, name in checks:
        if not check_fn():
            failed.append(name)
            
    print("\nValidation Summary")
    print("=" * 50)
    if failed:
        print(f"Failed checks: {', '.join(failed)}")
        print("\nPlease fix the above issues before proceeding with distillation")
        sys.exit(1)
    else:
        print("All checks passed! You can proceed with distillation")
        sys.exit(0)

if __name__ == "__main__":
    main()
