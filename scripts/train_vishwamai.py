#!/usr/bin/env python3
"""
Training script for VishwamAI models.

Example usage:
    python train_vishwamai.py --config configs/small_model.json
    python train_vishwamai.py --config configs/medium_model.json --use-lora
"""

import argparse
import jax
import jax.numpy as jnp
from pathlib import Path
import time

from vishwamai import (
    ModelConfig, TrainingConfig, CurriculumTrainer,
    VishwamAIModel, get_hardware_info, print_model_info,
    setup_mixed_precision, estimate_memory_usage
)


def create_dummy_dataset(batch_size: int, seq_len: int, vocab_size: int, num_batches: int = 100):
    """Create dummy dataset for demonstration."""
    
    def data_generator():
        rng = jax.random.PRNGKey(42)
        
        for _ in range(num_batches):
            rng, batch_rng = jax.random.split(rng)
            
            # Generate random token sequences
            input_ids = jax.random.randint(
                batch_rng,
                (batch_size, seq_len),
                minval=0,
                maxval=vocab_size
            )
            
            # Create attention mask (all ones for dummy data)
            attention_mask = jnp.ones((batch_size, seq_len), dtype=jnp.bool_)
            
            yield {
                'input_ids': input_ids,
                'attention_mask': attention_mask
            }
    
    return data_generator()


def main():
    parser = argparse.ArgumentParser(description='Train VishwamAI model')
    parser.add_argument('--config', type=str, required=True, help='Path to model config file')
    parser.add_argument('--use-lora', action='store_true', help='Use LoRA for parameter-efficient training')
    parser.add_argument('--batch-size', type=int, default=8, help='Training batch size')
    parser.add_argument('--seq-len', type=int, default=1024, help='Sequence length')
    parser.add_argument('--steps', type=int, default=1000, help='Number of training steps')
    parser.add_argument('--output-dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--dry-run', action='store_true', help='Just validate configuration without training')
    
    args = parser.parse_args()
    
    print("🚀 VishwamAI Training Script")
    print("=" * 50)
    
    # Print hardware information
    hardware_info = get_hardware_info()
    print(f"Hardware: {hardware_info['num_devices']} devices")
    print(f"Platforms: {set(hardware_info['device_types'])}")
    
    if hardware_info['has_gpu'] and isinstance(hardware_info.get('gpu_details'), list):
        for i, gpu in enumerate(hardware_info['gpu_details']):
            print(f"  GPU {i}: {gpu['name']} ({gpu['memory_total']}GB)")
    
    # Load or create model configuration
    if Path(args.config).exists():
        with open(args.config, 'r') as f:
            import json
            config_dict = json.load(f)
            model_config = ModelConfig(**config_dict)
    else:
        print(f"Config file {args.config} not found, using default small model config")
        model_config = ModelConfig(
            dim=1024,
            depth=12,
            heads=16,
            vocab_size=32000,
            max_seq_len=args.seq_len,
            use_flash_attention=True,
            use_grouped_query_attention=True,
            use_rmsnorm=True,
            gradient_checkpointing=True
        )
    
    # Print model information
    print_model_info(model_config)
    
    # Estimate memory usage
    memory_usage = estimate_memory_usage(model_config, args.batch_size, args.seq_len)
    print(f"\nMemory Estimates (batch_size={args.batch_size}, seq_len={args.seq_len}):")
    print(f"  Inference: {memory_usage['inference_gb']:.2f} GB")
    print(f"  Training: {memory_usage['total_gb']:.2f} GB")
    
    # Check if we have enough memory
    available_memory = 16.0  # Assume 16GB for demo
    if hardware_info['has_gpu'] and isinstance(hardware_info.get('gpu_details'), list):
        available_memory = min(gpu['memory_total'] for gpu in hardware_info['gpu_details'])
    
    if memory_usage['training_gb'] > available_memory * 0.8:  # Leave 20% headroom
        print(f"⚠️  Warning: Estimated memory usage ({memory_usage['training_gb']:.2f} GB) "
              f"exceeds available memory ({available_memory:.2f} GB)")
        print("Consider reducing batch size, sequence length, or using gradient checkpointing")
    
    if args.dry_run:
        print("✅ Configuration validated successfully!")
        return
    
    # Setup mixed precision
    mp_config = setup_mixed_precision()
    print(f"\nMixed Precision: {mp_config['use_mixed_precision']} ({mp_config['dtype']})")
    
    # Create training configuration
    training_config = TrainingConfig(
        model_config=model_config,
        batch_size=args.batch_size,
        max_seq_len=args.seq_len,
        total_steps=args.steps,
        use_lora=args.use_lora,
        checkpoint_dir=args.output_dir,
        use_bfloat16=mp_config['use_mixed_precision'] and mp_config['dtype'] == jnp.bfloat16,
        use_curriculum=True
    )
    
    # Create trainer
    trainer = CurriculumTrainer(training_config)
    
    # Initialize training state
    rng_key = jax.random.PRNGKey(42)
    state = trainer.initialize_state(rng_key)
    
    print(f"\nModel initialized with {trainer.model.config.vocab_size:,} vocabulary size")
    
    # Create dummy training data
    print("📊 Creating dummy training dataset...")
    train_loader = create_dummy_dataset(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=model_config.vocab_size,
        num_batches=args.steps
    )
    
    # Create dummy evaluation data
    eval_loader = create_dummy_dataset(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        vocab_size=model_config.vocab_size,
        num_batches=10
    )
    
    # Start training
    print(f"\n🏋️ Starting training for {args.steps} steps...")
    print(f"Curriculum stages: {len(training_config.curriculum_stages)}")
    
    start_time = time.time()
    
    try:
        trainer.train(
            train_loader=train_loader,
            eval_loader=eval_loader,
            rng_key=rng_key
        )
        
        end_time = time.time()
        print(f"\n✅ Training completed in {end_time - start_time:.2f} seconds")
        
        # Save final model
        output_path = Path(args.output_dir) / "final_model"
        trainer._save_checkpoint(args.steps)
        print(f"📁 Model saved to {output_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
        trainer._save_checkpoint(trainer.global_step)
        print(f"📁 Checkpoint saved at step {trainer.global_step}")
    
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()
