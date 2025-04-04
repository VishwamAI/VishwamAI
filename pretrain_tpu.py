"""Pre-training script optimized for TPU execution with Gemma distillation and thinking layers."""

import jax
import jax.numpy as jnp
from vishwamai.transformer import EnhancedTransformerModel, create_vishwamai_transformer
from vishwamai.pipeline import TPUDataPipeline, DistillationDataPipeline
from vishwamai.device_mesh import TPUMeshContext
from vishwamai.distill import create_student_model, initialize_from_teacher
from vishwamai.thoughts import TreeOfThoughts
import time
from tqdm.auto import tqdm

def main():
    # Load configuration
    config = get_pretrain_config()
    training_config = TPUTrainingConfig(
        model_config=config["model"],
        thinking_config=config["thinking"],  # Add thinking config
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

    # Initialize teacher model (Gemma-27B)
    teacher_model = create_vishwamai_transformer(
        model_name=config["distillation"]["teacher_model"],
        dtype=config["optimization"]["teacher_load_dtype"]
    )
    
    # Initialize Tree of Thoughts for teacher model
    teacher_tot = TreeOfThoughts(
        model=teacher_model,
        num_steps=config["thinking"]["num_steps"],
        max_branches=config["thinking"]["max_branches"],
        beam_width=config["thinking"]["beam_width"],
        temperature=config["thinking"]["temperature"]
    )
    
    # Create and initialize student model (7B) from teacher
    student_model, student_vars, student_config = create_student_model(
        config=config["model"],
        teacher_model=teacher_model,
        reduction_factor=32/80  # Ratio of student layers (32) to teacher layers (80)
    )
    
    # Initialize Tree of Thoughts for student model
    student_tot = TreeOfThoughts(
        model=student_model,
        num_steps=config["thinking"]["num_steps"],
        max_branches=config["thinking"]["max_branches"],
        beam_width=config["thinking"]["beam_width"],
        temperature=config["thinking"]["temperature"]
    )
    
    # Initialize distillation trainer with memory optimizations and thinking
    distill_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        student_config=student_config,
        temperature=config["distillation"]["temperature"],
        alpha=config["distillation"]["alpha"],
        use_flash_attn=config["model"]["use_flash_attn"],
        use_fp8=config["optimization"]["use_fp8_gemm"],
        block_size=config["optimization"]["block_size"],
        teacher_tot=teacher_tot,
        student_tot=student_tot
    )

    # Setup training state and data pipeline with profiling
    state, device_mesh, train_step, profiler = setup_tpu_training(
        training_config, 
        enable_profiling=True,
        model=student_model,
        initial_params=student_vars
    )

    # Initialize student parameters from teacher using uniform layer mapping
    state = initialize_from_teacher(
        student_state=state,
        teacher_state=teacher_model.params,
        method="layer_mapping",
        mapping_strategy=config["distillation"]["layer_mapping_strategy"]
    )

    # Create optimized data pipeline for distillation
    data_pipeline = DistillationDataPipeline(
        config=config, 
        teacher_model=teacher_model,
        devices=devices,
        enable_thinking=True  # Enable thinking in data pipeline
    )
    train_loader = data_pipeline.create_distillation_dataset(
        "train-*.parquet",
        is_training=True,
        cache_teacher_outputs=True
    )
    
    # Training loop with TPU optimizations
    with mesh_context:
        step = 0
        start_time = time.time()
        
        with tqdm(total=training_config.max_steps) as pbar:
            for batch in train_loader:
                if step >= training_config.max_steps:
                    break
                    
                # Training step with distillation and thinking
                state, metrics = train_step(state, batch)
                
                # Update progress
                step += 1
                pbar.update(1)
                pbar.set_postfix({
                    'loss': f"{metrics['loss']:.4f}",
                    'distill_loss': f"{metrics.get('distill_loss', 0.0):.4f}",
                    'thinking_loss': f"{metrics.get('thinking_loss', 0.0):.4f}",
                    'lr': f"{metrics.get('learning_rate', 0.0):.6f}",
                })
                
                # Log performance metrics and save checkpoints
                if step % 100 == 0 and profiler is not None:
                    stats = profiler.get_metrics_summary()
                    print(f"\nStep {step} statistics:")
                    print(f"TPU utilization: {stats['tpu_utilization_mean']:.2%}")
                    print(f"Memory usage: {stats['memory_usage_mean']:.2f} GB")
                    print(f"Training throughput: {stats['throughput_mean']:.2f} samples/sec")
                    print(f"Teacher agreement: {metrics.get('teacher_agreement', 0.0):.2%}")
                    print(f"Thinking quality: {metrics.get('thinking_quality', 0.0):.2%}")
                
                if step % config["training"]["checkpoint_steps"] == 0:
                    # Save checkpoint with thinking state
                    checkpoint_dict = {
                        'step': step,
                        'state': state,
                        'teacher_tot_state': teacher_tot.get_state(),
                        'student_tot_state': student_tot.get_state(),
                        'metrics': metrics
                    }
                    # Checkpoint saving logic here
        
        total_time = time.time() - start_time
        print(f"\nTraining completed in {total_time:.2f} seconds")
        print("Final training metrics:", metrics)

if __name__ == "__main__":
    main()