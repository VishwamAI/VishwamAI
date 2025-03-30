"""Pre-training script optimized for TPU execution."""

import jax
import jax.numpy as jnp
from vishwamai.transformer import EnhancedTransformerModel, create_vishwamai_transformer
from vishwamai.training import TPUTrainingConfig, create_train_state_tpu, setup_tpu_training
from vishwamai.pipeline import TPUDataPipeline
from vishwamai.device_mesh import TPUMeshContext
from configs.tpu_pretrain_config import get_pretrain_config
import time
from tqdm.auto import tqdm

def main():
    # Load configuration
    config = get_pretrain_config()
    training_config = TPUTrainingConfig(
        model_config=config["model"],
        batch_size=config["training"]["batch_size"],
        grad_accum_steps=config["training"]["grad_accum_steps"],
        learning_rate=config["training"]["learning_rate"],
        warmup_steps=config["training"]["warmup_steps"],
        max_steps=config["training"]["max_steps"],
        weight_decay=config["training"]["weight_decay"],
        max_grad_norm=config["training"]["max_grad_norm"],
        dtype=config["optimization"]["dtype"],
        enable_pjit=True,
        block_size=config["optimization"]["block_size"],
        use_flash_attn=config["model"]["use_flash_attn"],
        mixed_precision=config["optimization"]["mixed_precision"]
    )

    # Initialize TPU mesh
    devices = jax.devices()
    print(f"Available TPU devices: {devices}")
    mesh_context = TPUMeshContext(config, data_parallel=True)

    # Setup training state and data pipeline
    state, device_mesh, train_step, profiler = setup_tpu_training(
        training_config, 
        enable_profiling=True
    )

    # Create data pipeline
    data_pipeline = TPUDataPipeline(config, devices=devices)
    train_loader = data_pipeline.create_dataset("train-*.parquet", is_training=True)
    
    # Training loop
    with mesh_context:
        step = 0
        start_time = time.time()
        
        with tqdm(total=training_config.max_steps) as pbar:
            for batch in train_loader:
                if step >= training_config.max_steps:
                    break
                    
                # Training step
                state, metrics = train_step(state, batch)
                
                # Update progress
                step += 1
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'lr': f"{metrics.get('learning_rate', 0.0):.6f}",
                })
                
                # Log performance metrics every 100 steps
                if step % 100 == 0 and profiler is not None:
                    stats = profiler.get_metrics_summary()
                    print(f"\nStep {step} statistics:")
                    print(f"TPU utilization: {stats['tpu_utilization_mean']:.2%}")
                    print(f"Memory usage: {stats['memory_usage_mean']:.2f} GB")
                    print(f"Training throughput: {stats['throughput_mean']:.2f} samples/sec")
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print("Final training metrics:", metrics)

if __name__ == "__main__":
    main()