"""Verification script for QwQ distillation setup."""
import jax
import jax.numpy as jnp
from omegaconf import OmegaConf

from vishwamai.model import VishwamAIModel, ModelConfig
from vishwamai.qwen_distiller import QwenDistillationTrainer
from vishwamai.qwen_data import QwenDataLoader

def test_loader():
    """Test data loader with synthetic data."""
    print("\nTesting QwenDataLoader...")
    
    # Create synthetic safetensor files
    import os
    import numpy as np
    import safetensors.numpy as sf
    
    os.makedirs("test_data", exist_ok=True)
    for i in range(14):
        tensors = {
            'input_ids': np.random.randint(0, 100, (32, 512), dtype=np.int32),
            'attention_mask': np.ones((32, 512), dtype=np.int32)
        }
        sf.save_file(tensors, f"test_data/model-{i+1:05d}-of-00014.safetensors")
    
    # Test loader
    loader = QwenDataLoader(
        safetensor_dir="test_data",
        batch_size=1,
        gradient_accumulation_steps=16
    )
    
    print("Loader settings:")
    print(f"- Devices: {loader.num_devices}")
    print(f"- Global batch size: {loader.global_batch_size}")
    print(f"- Per-device batch size: {loader.per_device_batch_size}")
    
    # Test batch creation
    test_input = jnp.ones((16, 512), dtype=jnp.int32)
    batch = loader.create_training_batch(test_input)
    print(f"Batch shapes:")
    print(f"- input_ids: {batch['input_ids'].shape}")
    print(f"- attention_mask: {batch['attention_mask'].shape}")
    
    return True

def test_trainer():
    """Test trainer initialization and step."""
    print("\nTesting QwenDistillationTrainer...")
    
    # Create minimal config
    config = OmegaConf.create({
        'distillation': {
            'teacher_model': {
                'config': {
                    'hidden_size': 1024,
                    'num_layers': 4,
                    'num_attention_heads': 8
                }
            },
            'student_model': {
                'config': {
                    'hidden_size': 512,
                    'num_layers': 2,
                    'num_attention_heads': 4
                }
            }
        },
        'training': {
            'learning_rate': 1e-4,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 16
        },
        'distillation_params': {
            'temperature': 2.0,
            'alpha_ce': 0.2,
            'alpha_kd': 0.8,
            'feature_loss_weight': 0.1,
            'attention_loss_weight': 0.1
        }
    })
    
    # Initialize models
    teacher_model = VishwamAIModel(ModelConfig(**config.distillation.teacher_model.config))
    student_model = VishwamAIModel(ModelConfig(**config.distillation.student_model.config))
    
    # Initialize trainer
    trainer = QwenDistillationTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        cfg=config
    )
    
    # Test state creation
    rng = jax.random.PRNGKey(0)
    try:
        state = trainer.create_train_state(rng)
        print("Training state created successfully")
    except Exception as e:
        print(f"Failed to create training state: {e}")
        return False
    
    # Test training step with synthetic data
    batch = {
        'input_ids': jnp.ones((16, 512), dtype=jnp.int32),
        'attention_mask': jnp.ones((16, 512), dtype=jnp.int32)
    }
    
    try:
        state, metrics, grads = trainer.train_step_with_grads(state, batch, rng)
        print(f"Training step successful")
        print(f"Loss metrics: {metrics}")
    except Exception as e:
        print(f"Failed training step: {e}")
        return False
    
    return True

def main():
    """Run all tests."""
    print("Starting distillation setup verification...")
    print(f"JAX devices: {jax.devices()}")
    
    tests = [
        ("Data Loader", test_loader),
        ("Trainer", test_trainer)
    ]
    
    failed = []
    for name, test_fn in tests:
        print(f"\nTesting {name}...")
        try:
            if not test_fn():
                failed.append(name)
        except Exception as e:
            print(f"Test failed with error: {e}")
            failed.append(name)
    
    if failed:
        print(f"\nFailed tests: {', '.join(failed)}")
        return 1
    else:
        print("\nAll tests passed! Setup is ready for QwQ distillation.")
        return 0

if __name__ == "__main__":
    import sys
    sys.exit(main())
