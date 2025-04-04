"""Pre-training script optimized for TPU execution with Gemma distillation and thinking layers."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from vishwamai.model import VishwamAI, VishwamAIConfig
from vishwamai.transformer import EnhancedTransformerModel, TPUTrainingState 
from vishwamai.pipeline import TPUDataPipeline, DistillationDataPipeline
from vishwamai.device_mesh import TPUMeshContext
from vishwamai.distill import (
    create_student_model, 
    initialize_from_teacher,
    DistillationTrainer
)
from vishwamai.thoughts import TreeOfThoughts
from vishwamai.configs.tpu_v3_config import TPUV3Config
from vishwamai.configs.budget_model_config import BudgetModelConfig
from vishwamai.profiler import TPUProfiler
import time
from tqdm.auto import tqdm

@dataclass
class TPUTrainingConfig:
    """Training configuration for TPU."""
    model_config: Dict[str, Any]
    thinking_config: Dict[str, Any]
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    weight_decay: float
    max_grad_norm: float
    dtype: Any
    enable_pjit: bool = True
    block_size: int = 128
    use_flash_attn: bool = True
    mixed_precision: bool = True

def get_pretrain_config() -> Dict[str, Any]:
    """Get pre-training configuration."""
    # Start with TPU v3 optimized base config
    tpu_config = TPUV3Config()
    
    return {
        "model": {
            **tpu_config.model_config,
            "use_flash_attn": True,
            "vocab_size": 256000,  # Updated for Gemma-3 tokenizer
            "hidden_dim": 8192,    # Scaled up for 27B model
            "num_layers": 80,      # Gemma-3-27b architecture
            "num_heads": 64,       # Increased for 27B
            "head_dim": 128,
            "mlp_dim": 28672,     # Increased for 27B
            "max_seq_len": 8192,   # Support longer sequences
            "dropout_rate": 0.0,   # Reduced for pre-training
            "attention_dropout": 0.0
        },
        "training": {
            **tpu_config.training_config,
            "batch_size": 32,      # Adjusted for TPU memory
            "grad_accum_steps": 8, # Increased for effective batch size
            "learning_rate": 1e-4,
            "warmup_steps": 2000,
            "max_steps": 100000,
            "weight_decay": 0.01,
            "max_grad_norm": 1.0,
            "checkpoint_steps": 1000
        },
        "tpu": {
            **tpu_config.tpu_config,
            "device_strategy": "data_parallel"
        },
        "optimization": {
            "dtype": "bfloat16",
            "use_fp8_gemm": False,  # Disabled since fp8 not available
            "block_size": 128,
            "mixed_precision": True,
            "teacher_load_dtype": "bfloat16",
            "gradient_checkpointing": True  # Enable for memory efficiency
        },
        "thinking": {
            "num_steps": 3,
            "max_branches": 3,
            "max_depth": 5,
            "beam_width": 5,
            "temperature": 0.7
        },
        "distillation": {
            "teacher_model": "google/gemma-3-27b-pt",  # Updated to 27B
            "temperature": 2.0,
            "alpha": 0.5,
            "layer_mapping_strategy": "uniform"
        }
    }

def create_vishwamai_transformer(model_name: str, dtype: str = "bfloat16") -> Tuple[VishwamAI, Any]:
    """Create a VishwamAI transformer model and initialize its parameters."""
    config = VishwamAIConfig(
        vocab_size=32000,
        hidden_dim=2048,
        num_layers=24,
        num_heads=16, 
        head_dim=128,
        mlp_dim=8192,
        max_seq_len=2048,
        dropout_rate=0.1,
        attention_dropout=0.1,
        use_flash_attn=True
    )
    
    model = VishwamAI(config=config)
    
    # Initialize model parameters
    rng = jax.random.PRNGKey(0)
    dummy_input = jnp.ones((1, config.max_seq_len), dtype=jnp.int32)
    variables = model.init(rng, dummy_input)
    
    return model, variables['params']

def setup_tpu_training(
    config: TPUTrainingConfig,
    enable_profiling: bool = True,
    model: Optional[EnhancedTransformerModel] = None,
    initial_params: Optional[Dict[str, Any]] = None
) -> Tuple[TPUTrainingState, Any, Any, Optional[TPUProfiler]]:
    """Set up TPU training state and components."""
    
    # Create device mesh
    devices = jax.devices()
    device_mesh = jax.sharding.Mesh(devices, ("data",))
    
    # Initialize model if not provided
    if model is None:
        model = create_vishwamai_transformer(config.model_config)[0]
    
    # Set up training state
    rng = jax.random.PRNGKey(42)
    state = TPUTrainingState(
        params=initial_params or model.init(rng, jnp.ones((1, 128), dtype=jnp.int32)),
        opt_state=None,  # Will be initialized by optimizer
        model_fn=model.apply,
        tx=None  # Will be initialized by optimizer
    )
    
    # Create profiler if requested
    profiler = TPUProfiler(config) if enable_profiling else None
    
    # Create training step function
    @jax.jit
    def train_step(state: TPUTrainingState, batch: Dict[str, jnp.ndarray]):
        """Single training step."""
        def loss_fn(params):
            logits = state.model_fn({"params": params}, batch["input_ids"])
            return jax.nn.softmax_cross_entropy_with_integer_labels(
                logits, batch["labels"]
            ).mean()
        
        grad_fn = jax.value_and_grad(loss_fn)
        loss, grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        
        metrics = {"loss": loss}
        return new_state, metrics
    
    return state, device_mesh, train_step, profiler

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

    # Initialize teacher model (Gemma-27B) with parameters
    teacher_model, teacher_params = create_vishwamai_transformer(
        model_name=config["distillation"]["teacher_model"],
        dtype=config["optimization"]["teacher_load_dtype"]
    )
    
    # Initialize Tree of Thoughts for teacher model
    teacher_tot = TreeOfThoughts(
        model=teacher_model,
        params=teacher_params,  # Use initialized parameters
        tokenizer=teacher_model.tokenizer,
        max_branches=config["thinking"]["max_branches"],
        max_depth=config["thinking"]["max_depth"],
        beam_width=config["thinking"]["beam_width"],
        temperature=config["thinking"]["temperature"]
    )
    
    # Create and initialize student model from teacher with memory optimizations
    student_model, student_vars, student_config = create_student_model(
        {
            **config["model"],
            "gradient_checkpointing": True,  # Enable gradient checkpointing
            "hidden_dim": 4096,    # Reduce model size initially
            "num_layers": 24,      # Reduce number of layers
            "num_heads": 32,       # Adjust head count
            "mlp_dim": 16384      # Reduce MLP dimension
        },
        teacher_model,
        24/80,  # Adjusted reduction factor for smaller initial model
        jax.random.PRNGKey(42)
    )
    
    # Initialize Tree of Thoughts for student model 
    student_tot = TreeOfThoughts(
        model=student_model,
        params=student_vars,
        tokenizer=student_model.tokenizer,
        max_branches=config["thinking"]["max_branches"],
        max_depth=config["thinking"]["max_depth"],
        beam_width=config["thinking"]["beam_width"],
        temperature=config["thinking"]["temperature"]
    )
    
    # Initialize distillation trainer with memory optimizations and thinking
    distill_trainer = DistillationTrainer(
        teacher_model=teacher_model,
        teacher_params=teacher_params,  # Pass teacher parameters
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
        teacher_state=teacher_params,
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