"""Training module for VishwamAI transformer with enhanced TPU support."""

import jax
import jax.numpy as jnp
import optax
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Callable, Iterator
from jax.sharding import PartitionSpec as P
from tqdm.auto import tqdm
from vishwamai.pipeline import VishwamAIPipeline
from vishwamai.profiler import TPUProfiler
from vishwamai.transformer import create_learning_rate_schedule
from vishwamai.logger import DuckDBLogger
from jax.experimental import pjit  # Updated import path for pjit

"""Training configuration and initialization"""

import flax
from flax.training import train_state

from vishwamai.transformer import EnhancedTransformerModel
from vishwamai.distill import DistillationTrainer

@dataclass
class TPUTrainingConfig:  # Renamed from TrainingConfig to TPUTrainingConfig
    """Training configuration with proper typing"""
    model_config: Dict[str, Any]
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    weight_decay: float
    max_grad_norm: float
    dtype: str
    enable_pjit: bool
    block_size: int
    use_flash_attn: bool
    mixed_precision: bool

def create_training_config() -> TPUTrainingConfig:
    """Create TPU-optimized training configuration for 13B distillation"""
    model_config = {
        'vocab_size': 32000,
        'num_layers': 40,     # Scaled up for 13B
        'num_heads': 24,      # Increased head count
        'head_dim': 128,      # Optimized for v5e
        'hidden_dim': 3072,   # 13B architecture
        'mlp_dim': 12288,     # 4x hidden_dim
        'max_seq_len': 2048,
        'dropout_rate': 0.1,
        'use_flash_attn': True,
        'use_rotary': True,
        'use_rms_norm': True  # Better stability
    }
    
    return TPUTrainingConfig(
        model_config=model_config,
        batch_size=64,        # Increased for v5e
        grad_accum_steps=8,   # Accumulate for effective 512 batch
        learning_rate=1e-4,
        warmup_steps=2000,
        max_steps=150000,     # Longer training for 13B
        weight_decay=0.01,
        max_grad_norm=1.0,
        dtype='bfloat16',     # TPU v5e optimal
        enable_pjit=True,
        block_size=128,       # Optimal chunk size
        use_flash_attn=True,
        mixed_precision=True  # Enable mixed precision
    )

def create_train_state_tpu(
    config: TPUTrainingConfig,
    rng: Any,
    mesh: Optional[Any] = None,
    profiler: Optional[TPUProfiler] = None
) -> Any:
    """Create training state optimized for TPU execution."""
    
    # Create learning rate schedule
    lr_schedule = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=config.learning_rate,
        warmup_steps=config.warmup_steps,
        decay_steps=config.max_steps,
        end_value=0.0
    )
    
    # Configure optimizer with TPU optimizations
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.95,
            eps=1e-8,
            weight_decay=config.weight_decay
        )
    )
    
    # Profile model initialization if profiler is available
    if profiler:
        with profiler.profile_region("model_init"):
            model = EnhancedTransformerModel(
                vocab_size=config.model_config['vocab_size'],
                num_layers=config.model_config['num_layers'],
                num_heads=config.model_config['num_heads'],
                head_dim=config.model_config['head_dim'],
                hidden_dim=config.model_config['hidden_dim'],
                mlp_dim=config.model_config['mlp_dim'],
                max_seq_len=config.model_config['max_seq_len'],
                dropout_rate=config.model_config.get('dropout_rate', 0.1),
                use_flash_attn=config.model_config.get('use_flash_attn', config.use_flash_attn),
                use_rms_norm=config.model_config.get('use_rms_norm', False),
                dtype=config.dtype
            )
            variables = model.init(rng, jnp.ones((2, config.block_size), dtype=jnp.int32))
    else:
        model = EnhancedTransformerModel(
            vocab_size=config.model_config['vocab_size'],
            num_layers=config.model_config['num_layers'],
            num_heads=config.model_config['num_heads'],
            head_dim=config.model_config['head_dim'],
            hidden_dim=config.model_config['hidden_dim'],
            mlp_dim=config.model_config['mlp_dim'],
            max_seq_len=config.model_config['max_seq_len'],
            dropout_rate=config.model_config.get('dropout_rate', 0.1),
            use_flash_attn=config.model_config.get('use_flash_attn', config.use_flash_attn),
            use_rms_norm=config.model_config.get('use_rms_norm', False),
            dtype=config.dtype
        )
        variables = model.init(rng, jnp.ones((2, config.block_size), dtype=jnp.int32))
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )

def create_train_step_tpu(
    config: TPUTrainingConfig,
    state: Any,
    profiler: Optional[TPUProfiler] = None
) -> Callable:
    """Create TPU-optimized training step function with profiling."""
    
    def train_step(
        state: Any,
        batch: Dict[str, jnp.ndarray],
        dropout_rng: Any
    ) -> Tuple[Any, Dict[str, float]]:
        
        def loss_fn(params):
            logits = state.apply_fn(
                {'params': params},
                batch['input_ids'],
                deterministic=False,
                rngs={'dropout': dropout_rng}
            )
            return optax.softmax_cross_entropy_with_integer_labels(
                logits=logits,
                labels=batch['labels']
            ).mean()
        
        if profiler:
            with profiler.profile_region("forward_backward"):
                grad_fn = jax.value_and_grad(loss_fn)
                loss, grads = grad_fn(state.params)
        else:
            grad_fn = jax.value_and_grad(loss_fn)
            loss, grads = grad_fn(state.params)
        
        # Update state
        new_state = state.apply_gradients(grads=grads)
        metrics = {'loss': loss}
        
        # Record metrics if profiling
        if profiler:
            profiler.record_memory(loss_fn)
            profiler.record_flops(loss_fn)
        
        return new_state, metrics
    
    # Enable pjit with mesh sharding if configured
    if config.enable_pjit:
        train_step = pjit(
            train_step,
            in_axis_resources=(None, P('data'), None),
            out_axis_resources=(None, None),
            donate_argnums=(0,)
        )
    else:
        train_step = jax.jit(train_step)
    
    return train_step

def create_eval_step_tpu(
    config: TPUTrainingConfig,
    state: Any
) -> Callable:
    """Create TPU-optimized evaluation step function."""
    
    def eval_step(
        params: Any,
        batch: Dict[str, jnp.ndarray]
    ) -> Dict[str, float]:
        # Process in chunks to handle long sequences
        logits_chunks = []
        for i in range(0, batch['input_ids'].shape[1], config.block_size):
            chunk = jax.lax.dynamic_slice(
                batch['input_ids'],
                (0, i),
                (batch['input_ids'].shape[0], 
                 min(config.block_size, batch['input_ids'].shape[1] - i))
            )
            chunk_logits = state.apply_fn(
                {'params': params},
                chunk,
                deterministic=True
            )
            logits_chunks.append(chunk_logits)
            
        logits = jnp.concatenate(logits_chunks, axis=1)
        
        # Compute metrics
        loss = optax.softmax_cross_entropy_with_integer_labels(
            logits=logits,
            labels=batch['labels']
        ).mean()
        
        accuracy = jnp.mean(
            jnp.argmax(logits, axis=-1) == batch['labels']
        )
        
        return {
            'loss': loss,
            'accuracy': accuracy,
            'perplexity': jnp.exp(loss)
        }
    
    if config.enable_pjit:
        eval_step = pjit(
            eval_step,
            in_axis_resources=None,
            out_axis_resources=None
        )
    else:
        eval_step = jax.jit(eval_step)
        
    return eval_step

def setup_tpu_training(
    config: TPUTrainingConfig,
    seed: int = 42,
    enable_profiling: bool = True
) -> Tuple[Any, Any, Callable, TPUProfiler]:
    """Set up TPU training with integrated profiling."""
    
    # Initialize profiler if enabled
    profiler = TPUProfiler(config.model_config) if enable_profiling else None
    
    # Set up TPU device mesh
    device_mesh = None
    if config.enable_pjit:
        devices = jax.devices()
        mesh_shape = (len(devices),)
        device_mesh = jax.sharding.Mesh(devices, ('data',))
    
    # Initialize training state
    rng = jax.random.PRNGKey(seed)
    state = create_train_state_tpu(config, rng, device_mesh, profiler)
    train_step = create_train_step_tpu(config, state, profiler)
    
    return state, device_mesh, train_step, profiler

def get_tpu_compile_options(
    config: TPUTrainingConfig
) -> Dict[str, Any]:
    """Get XLA compilation options optimized for TPU."""
    return {
        "num_partitions": 1,
        "enable_xla": True,
        "enable_checkpointing": True,
        "preserve_host_calls": True,
        "parameter_formation": "xla",
        "xla_shape_checks": "error",
        "allow_host_callbacks": True,
        "xla_gpu_autotune_level": 0,  # Disable for TPU
    }

class VishwamAITrainer:
    """Training manager for VishwamAI models."""
    
    def __init__(
        self,
        pipeline: VishwamAIPipeline,
        config: Dict[str, Any],
        train_loader: Iterator,
        eval_loader: Optional[Iterator] = None,
        experiment_name: Optional[str] = None,
        db_path: str = "training_logs.db"
    ):
        self.pipeline = pipeline
        self.config = config
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        
        # Training state
        self.current_step = 0
        self.best_eval_loss = float('inf')
        self.early_stopping_counter = 0
        
        # Initialize learning rate schedule
        self.lr_schedule = create_learning_rate_schedule(
            base_learning_rate=config['training']['learning_rate'],
            warmup_steps=config['training']['warmup_steps'],
            decay_steps=config['training']['train_steps']
        )
        
        # Setup training
        self.pipeline.setup_training(self.lr_schedule)
        
        # Initialize DuckDB logger
        self.logger = DuckDBLogger(
            db_path=db_path,
            experiment_name=experiment_name,
            config=config
        )
    
    def train(self):
        """Run the training loop."""
        print("Starting training...")
        total_steps = self.config['training']['train_steps']
        
        try:
            for step in tqdm(range(self.current_step, total_steps)):
                self.current_step = step
                
                # Get next batch
                try:
                    batch = next(self.train_loader)
                except StopIteration:
                    print("Reached end of dataset, restarting...")
                    continue
                
                # Training step
                metrics = self._train_step(batch)
                
                # Log metrics
                self._log_metrics(metrics, step)
                
                # Evaluate and save checkpoint if needed
                if step % self.config['logging']['eval_every'] == 0:
                    self._run_evaluation(step)
                
                # Save checkpoint
                if step % self.config['logging']['save_every'] == 0:
                    self._save_checkpoint(step)
                
                # Check for early stopping
                if self._should_stop():
                    print("Early stopping triggered")
                    break
                    
        finally:
            # Export final logs and close logger
            self._export_final_logs()
            self.logger.close()
    
    def _train_step(self, batch: Dict[str, jnp.ndarray]) -> Dict[str, float]:
        """Execute single training step."""
        dropout_rng = jax.random.PRNGKey(int(time.time()))
        
        # Perform training step
        new_state, metrics = self.pipeline.train_step(
            self.pipeline.state,
            batch,
            dropout_rng
        )
        
        # Update state
        self.pipeline.state = new_state
        
        return metrics
    
    def _run_evaluation(self, step: int):
        """Run evaluation loop."""
        if self.eval_loader is None:
            return
            
        eval_metrics = []
        for _ in range(self.config.get('eval_steps', 100)):
            try:
                batch = next(self.eval_loader)
            except StopIteration:
                break
                
            # Get evaluation metrics
            metrics = self.pipeline.eval_step(
                self.pipeline.state,
                batch
            )
            eval_metrics.append(metrics)
        
        # Compute average metrics
        avg_metrics = {
            k: jnp.mean([m[k] for m in eval_metrics])
            for k in eval_metrics[0].keys()
        }
        
        # Log evaluation metrics
        self._log_metrics(avg_metrics, step, prefix='eval')
        
        # Update best loss and early stopping
        if avg_metrics['loss'] < self.best_eval_loss:
            self.best_eval_loss = avg_metrics['loss']
            self.early_stopping_counter = 0
            
            # Save best model
            self._save_checkpoint(step, is_best=True)
        else:
            self.early_stopping_counter += 1
    
    def _log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = 'train'
    ):
        """Log metrics using DuckDB logger."""
        # Format metrics for logging
        log_metrics = {
            k: float(v)
            for k, v in metrics.items()
        }
        
        # Log to DuckDB
        self.logger.log_metrics(log_metrics, step, prefix)
        
        # Print to console
        if step % self.config['logging']['log_every'] == 0:
            metrics_str = " ".join(
                f"{k}: {v:.4f}"
                for k, v in log_metrics.items()
            )
            print(f"Step {step}: {metrics_str}")
    
    def _save_checkpoint(self, step: int, is_best: bool = False):
        """Save model checkpoint."""
        checkpoint_dir = self.config['checkpoint_dir']
        
        # Save current checkpoint
        checkpoint_path = f"{checkpoint_dir}/step_{step}.ckpt"
        self.pipeline.save_checkpoint(checkpoint_path)
        
        # Save best model if needed
        if is_best:
            best_path = f"{checkpoint_dir}/best_model.ckpt"
            self.pipeline.save_checkpoint(best_path)
    
    def _should_stop(self) -> bool:
        """Check if training should stop early."""
        patience = self.config.get('early_stopping_patience', 5)
        return self.early_stopping_counter >= patience
    
    def _export_final_logs(self):
        """Export final training logs and summary."""
        # Export logs to CSV
        self.logger.export_to_csv()
        
        # Get and print experiment summary
        summary = self.logger.get_experiment_summary()
        print("\nTraining Summary:")
        print(f"Total Steps: {summary['total_steps']}")
        print(f"Duration: {summary['end_time'] - summary['start_time']}")
        print("\nMetrics Summary:")
        for metric, stats in summary['metrics_summary'].items():
            print(f"{metric}:")
            print(f"  Min: {stats['min']:.4f}")
            print(f"  Max: {stats['max']:.4f}")
            print(f"  Mean: {stats['mean']:.4f}")
            print(f"  Std: {stats['std']:.4f}")
    
    def load_checkpoint(self, path: str):
        """Load training checkpoint."""
        self.pipeline.load_checkpoint(path)

def create_trainer(
    config: Dict[str, Any],
    train_loader: Iterator,
    eval_loader: Optional[Iterator] = None,
    tokenizer: Optional[Any] = None,
    model: Optional[Any] = None,
    teacher_model: Optional[Any] = None,
    experiment_name: Optional[str] = None,
    db_path: str = "training_logs.db"
) -> VishwamAITrainer:
    """Create a trainer instance with all components."""
    
    # Create pipeline
    pipeline = VishwamAIPipeline(
        config=config,
        tokenizer=tokenizer,
        model=model,
        teacher_model=teacher_model
    )
    
    # Create and return trainer
    return VishwamAITrainer(
        pipeline=pipeline,
        config=config,
        train_loader=train_loader,
        eval_loader=eval_loader,
        experiment_name=experiment_name,
        db_path=db_path
    )
