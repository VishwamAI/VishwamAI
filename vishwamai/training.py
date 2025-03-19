"""
Training module for VishwamAI transformer.
"""

import jax
import jax.numpy as jnp
import optax
import time
from typing import Any, Dict, Optional, Tuple, Callable, Iterator
from jax.sharding import PartitionSpec as P
from tqdm.auto import tqdm
from vishwamai.pipeline import VishwamAIPipeline
from vishwamai.transformer import create_learning_rate_schedule
from vishwamai.logger import DuckDBLogger

"""TPU-optimized training configuration and initialization"""

import flax
from flax.training import train_state

from vishwamai.transformer import EnhancedTransformerModel
from vishwamai.distill import DistillationTrainer
from dataclasses import dataclass

@dataclass
class TPUTrainingConfig:
    """Configuration for TPU-optimized training."""
    model_config: Dict[str, Any]
    batch_size: int
    grad_accum_steps: int
    learning_rate: float
    warmup_steps: int
    max_steps: int
    weight_decay: float
    max_grad_norm: float
    dtype: str = 'bfloat16'
    enable_pjit: bool = True
    block_size: int = 128
    use_flash_attn: bool = True
    mixed_precision: bool = True
    data_parallel: bool = True
    model_parallel: bool = True
    pipeline_parallel: bool = False
    pipeline_stages: int = 2  # For 2x2x2 topology
    model_parallel_size: int = 2  # For 2x2x2 topology

class MultimodalTrainingConfig(TPUTrainingConfig):
    """Configuration for multimodal training"""
    def __init__(
        self,
        model_config: Dict[str, Any],
        vision_config: Optional[Dict[str, Any]] = None,
        contrastive_loss_weight: float = 0.5,
        image_text_alignment_loss_weight: float = 0.3,
        **kwargs
    ):
        super().__init__(model_config, **kwargs)
        self.vision_config = vision_config
        self.contrastive_loss_weight = contrastive_loss_weight
        self.image_text_alignment_loss_weight = image_text_alignment_loss_weight

def create_train_state_tpu(
    config: TPUTrainingConfig,
    rng: Any,
    mesh: Optional[Any] = None
) -> Any:
    """
    Create training state optimized for TPU execution.
    
    Args:
        config: Training configuration
        rng: PRNG key
        mesh: Optional device mesh for model parallelism
    """
    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps
    )
    decay_fn = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.max_steps - config.warmup_steps
    )
    lr_schedule = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[config.warmup_steps]
    )
    
    # Create optimizer with TPU-optimized gradient transforms
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adamw(
            learning_rate=lr_schedule,
            b1=0.9,
            b2=0.95,  # Increased for better TPU stability
            eps=1e-8,
            weight_decay=config.weight_decay
        )
    )

    # Create model with TPU optimizations
    model = EnhancedTransformerModel(
        vocab_size=config.model_config['vocab_size'],
        num_layers=config.model_config['num_layers'],
        num_heads=config.model_config['num_heads'],
        head_dim=config.model_config['head_dim'],
        hidden_dim=config.model_config['hidden_dim'],
        mlp_dim=config.model_config['mlp_dim'],
        max_seq_len=config.model_config['max_seq_len'],
        dropout_rate=config.model_config.get('dropout_rate', 0.1),
        use_flash_attn=config.use_flash_attn,
        use_rms_norm=True,
        dtype=config.dtype
    )
    
    # Initialize parameters with optimal TPU layout
    dummy_input = jnp.ones((2, config.model_config['max_seq_len']), dtype=jnp.int32)
    variables = model.init(rng, dummy_input, deterministic=False)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )

def create_multimodal_train_state(
    config: MultimodalTrainingConfig,
    rng: Any,
    mesh: Optional[Any] = None
) -> Any:
    """Create training state for multimodal model"""
    # Add vision config to model config
    model_config = config.model_config.copy()
    model_config['vision_config'] = config.vision_config
    
    # Create learning rate schedule
    warmup_fn = optax.linear_schedule(
        init_value=0.0,
        end_value=config.learning_rate,
        transition_steps=config.warmup_steps
    )
    decay_fn = optax.cosine_decay_schedule(
        init_value=config.learning_rate,
        decay_steps=config.max_steps - config.warmup_steps
    )
    lr_schedule = optax.join_schedules(
        schedules=[warmup_fn, decay_fn],
        boundaries=[config.warmup_steps]
    )
    
    # TPU-optimized optimizer chain
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
    
    # Create model with vision encoder
    model = EnhancedTransformerModel(
        vocab_size=model_config['vocab_size'],
        num_layers=model_config['num_layers'],
        num_heads=model_config['num_heads'],
        head_dim=model_config['head_dim'],
        hidden_dim=model_config['hidden_dim'],
        mlp_dim=model_config['mlp_dim'],
        max_seq_len=model_config['max_seq_len'],
        dropout_rate=model_config.get('dropout_rate', 0.1),
        vision_config=config.vision_config,
        dtype=config.dtype
    )
    
    # Initialize with dummy inputs
    text_input = jnp.ones((2, model_config['max_seq_len']), dtype=jnp.int32)
    image_input = None
    if config.vision_config:
        image_size = config.vision_config.get('image_size', 896)
        image_input = jnp.ones((2, image_size, image_size, 3), dtype=config.dtype)
    
    variables = model.init(rng, text_input, image_input, deterministic=False)
    
    return train_state.TrainState.create(
        apply_fn=model.apply,
        params=variables['params'],
        tx=optimizer
    )

def create_train_step_tpu(
    config: TPUTrainingConfig,
    state: Any,
    trainer: Optional[DistillationTrainer] = None
) -> Callable:
    """Create TPU-optimized training step function."""
    def train_step(
        state: Any,
        batch: Dict[str, jnp.ndarray],
        dropout_rng: Any
    ) -> Tuple[Any, Dict[str, float]]:
        # Split batch for pipeline stages
        micro_batch_size = batch['input_ids'].shape[0] // (config.grad_accum_steps * 2)  # 2 pipeline stages
        
        def compute_loss(params, chunk):
            outputs = state.apply_fn(
                {'params': params},
                chunk['input_ids'],
                deterministic=False,
                rngs={'dropout': dropout_rng}
            )
            
            if trainer is not None:
                return trainer.compute_distillation_loss(
                    outputs,
                    chunk['teacher_logits'],
                    chunk['labels'],
                    chunk.get('attention_mask')
                )
            else:
                return optax.softmax_cross_entropy_with_integer_labels(
                    logits=outputs,
                    labels=chunk['labels']
                ).mean()
        
        # Gradient function with automatic device sharding
        grad_fn = jax.value_and_grad(compute_loss)
        
        # Initialize accumulators
        total_loss = 0.0
        total_grads = None
        
        # Accumulate gradients across micro-batches
        for i in range(config.grad_accum_steps):
            for j in range(2):  # Pipeline stages
                start_idx = (i * 2 + j) * micro_batch_size
                end_idx = start_idx + micro_batch_size
                
                chunk = {
                    k: v[start_idx:end_idx]
                    for k, v in batch.items()
                }
                
                # Compute gradients
                loss, grads = grad_fn(state.params, chunk)
                total_loss += loss
                
                if total_grads is None:
                    total_grads = grads
                else:
                    total_grads = jax.tree_map(
                        lambda x, y: x + y,
                        total_grads,
                        grads
                    )
        
        # Average gradients
        total_grads = jax.tree_map(
            lambda x: x / (config.grad_accum_steps * 2),
            total_grads
        )
        
        # Update state with synchronized gradients
        new_state = state.apply_gradients(grads=total_grads)
        metrics = {'loss': total_loss / (config.grad_accum_steps * 2)}
        
        return new_state, metrics
    
    # Enable pjit with mesh sharding
    if config.enable_pjit:
        mesh_axis_names = ('data', 'model', 'pipe') if config.pipeline_parallel else ('data', 'model')
        train_step = jax.pjit(
            train_step,
            in_shardings=(
                None,  # state
                P('data', None),  # batch
                None,  # dropout_rng
            ),
            out_shardings=(
                None,  # new_state
                None,  # metrics
            ),
            donate_argnums=(0,)  # Allow state buffer reuse
        )
    else:
        train_step = jax.jit(train_step)
    
    return train_step

def create_multimodal_train_step(
    config: MultimodalTrainingConfig,
    state: Any,
    trainer: Optional[DistillationTrainer] = None
) -> Callable:
    """Create training step function for multimodal model"""
    
    def compute_contrastive_loss(image_features, text_features, temperature=1.0):
        # Normalize features
        image_features = image_features / jnp.linalg.norm(image_features, axis=-1, keepdims=True)
        text_features = text_features / jnp.linalg.norm(text_features, axis=-1, keepdims=True)
        
        # Compute similarity matrix
        logits = jnp.einsum('bd,nd->bn', text_features, image_features) * temperature
        
        # Contrastive loss in both directions
        labels = jnp.arange(len(logits))
        loss_i2t = optax.softmax_cross_entropy(logits, labels)
        loss_t2i = optax.softmax_cross_entropy(logits.T, labels)
        
        return (loss_i2t + loss_t2i) / 2.0
    
    def compute_alignment_loss(image_features, text_features, attention_mask=None):
        # Compute alignment between image and text features
        alignment_scores = jnp.einsum('bld,bmd->blm', text_features, image_features)
        if attention_mask is not None:
            alignment_scores = jnp.where(attention_mask[..., None], alignment_scores, -1e9)
        
        # Bidirectional alignment loss
        text2image = jnp.mean(jax.nn.softmax(alignment_scores, axis=-1), axis=1)
        image2text = jnp.mean(jax.nn.softmax(alignment_scores, axis=1), axis=1)
        
        return -jnp.mean(jnp.log(text2image + 1e-9) + jnp.log(image2text + 1e-9))
    
    def train_step(
        state: Any,
        batch: Dict[str, jnp.ndarray],
        dropout_rng: Any
    ) -> Tuple[Any, Dict[str, float]]:
        
        def compute_loss(params, batch):
            # Standard text loss
            outputs = state.apply_fn(
                {'params': params},
                batch['input_ids'],
                batch.get('image_input'),
                deterministic=False,
                rngs={'dropout': dropout_rng}
            )
            
            if trainer is not None:
                text_loss = trainer.compute_distillation_loss(
                    outputs['text_logits'],
                    batch['teacher_logits'],
                    batch['labels'],
                    batch.get('attention_mask')
                )
            else:
                text_loss = optax.softmax_cross_entropy_with_integer_labels(
                    logits=outputs['text_logits'],
                    labels=batch['labels']
                ).mean()
            
            total_loss = text_loss
            loss_dict = {'text_loss': text_loss}
            
            # Add multimodal losses if image input present
            if 'image_input' in batch:
                # Contrastive loss
                contrastive_loss = compute_contrastive_loss(
                    outputs['image_features'],
                    outputs['text_features']
                )
                loss_dict['contrastive_loss'] = contrastive_loss
                total_loss += config.contrastive_loss_weight * contrastive_loss
                
                # Alignment loss
                alignment_loss = compute_alignment_loss(
                    outputs['image_features'],
                    outputs['text_features'],
                    batch.get('attention_mask')
                )
                loss_dict['alignment_loss'] = alignment_loss
                total_loss += config.image_text_alignment_loss_weight * alignment_loss
            
            loss_dict['total_loss'] = total_loss
            return total_loss, loss_dict
        
        # Accumulate gradients across chunks
        grad_fn = jax.value_and_grad(compute_loss, has_aux=True)
        (total_loss, loss_dict), grads = grad_fn(state.params, batch)
        
        # Update state
        new_state = state.apply_gradients(grads=grads)
        
        return new_state, loss_dict
    
    # Enable pjit for TPU if configured
    if config.enable_pjit:
        train_step = jax.pjit(
            train_step,
            in_axis_resources=None,
            out_axis_resources=None
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
        eval_step = jax.pjit(
            eval_step,
            in_axis_resources=None,
            out_axis_resources=None
        )
    else:
        eval_step = jax.jit(eval_step)
        
    return eval_step

def setup_tpu_training(
    config: TPUTrainingConfig,
    seed: int = 42
) -> Tuple[Any, Any, Callable, Callable]:
    """
    Set up complete TPU training pipeline.
    
    Args:
        config: Training configuration
        seed: Random seed
    
    Returns:
        Tuple containing:
        - Training state
        - Device mesh
        - Training step function
        - Evaluation step function
    """
    # Set up device mesh for TPU
    devices = jax.devices()
    mesh_shape = (len(devices),)
    device_mesh = jax.sharding.Mesh(devices, ('batch',))
    
    # Initialize random keys
    rng = jax.random.PRNGKey(seed)
    rng, init_rng = jax.random.split(rng)
    
    # Create training state
    with device_mesh:
        state = create_train_state_tpu(config, init_rng, device_mesh)
        
    # Create step functions
    train_step_fn = create_train_step_tpu(config, state)
    eval_step_fn = create_eval_step_tpu(config, state)
    
    return state, device_mesh, train_step_fn, eval_step_fn

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

