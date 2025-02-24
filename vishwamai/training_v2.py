#!/usr/bin/env python3
import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import hydra
import jax
import jax.numpy as jnp
import optuna
from flax.training import train_state, checkpoints, dynamic_scale
from flax.jax_utils import replicate, unreplicate
from hydra.utils import instantiate
from omegaconf import DictConfig
import wandb
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from .model import VishwamAIModel, ModelConfig, create_optimizer
from .tokenizer import VishwamAITokenizer
from .error_correction import ModelIntegrator
from .tot import TreeOfThoughts
from .data_utils import create_train_dataloader, create_val_dataloader, evaluate
from .distillation import DistillationTrainer

logger = logging.getLogger(__name__)

def setup_monitoring(cfg: DictConfig) -> Tuple[Optional[SummaryWriter], Optional[wandb.Run]]:
    """Initialize monitoring tools based on configuration."""
    tb_writer = None
    wandb_run = None
    
    if cfg.monitoring.enabled:
        if cfg.monitoring.tensorboard.enabled:
            tb_writer = SummaryWriter(cfg.monitoring.tensorboard.log_dir)
        
        if cfg.monitoring.wandb.enabled:
            wandb_run = wandb.init(
                project=cfg.monitoring.wandb.project,
                name=cfg.monitoring.wandb.name,
                config=dict(cfg),
                resume="allow"
            )
    
    return tb_writer, wandb_run

def create_train_state_with_amp(
    model: VishwamAIModel,
    cfg: DictConfig,
    rng: jax.random.PRNGKey
) -> train_state.TrainState:
    """Create training state with automatic mixed precision support."""
    params_rng, dropout_rng = jax.random.split(rng)
    
    sample_input = jnp.ones((1, cfg.model.max_seq_len), dtype=jnp.int32)
    params = model.init(
        {'params': params_rng, 'dropout': dropout_rng},
        input_ids=sample_input,
        train=True
    )['params']
    
    # Initialize optimizer with learning rate schedule
    optimizer = create_optimizer(
        learning_rate=cfg.training.optimizer.lr,
        weight_decay=cfg.training.optimizer.weight_decay,
        beta1=cfg.training.optimizer.beta1,
        beta2=cfg.training.optimizer.beta2,
        warmup_steps=cfg.training.warmup_steps,
        total_steps=cfg.training.max_steps,
        clip_norm=cfg.training.optimizer.clip_grad_norm
    )
    
    # Setup automatic mixed precision if enabled
    dynamic_scale_instance = None
    if cfg.training.amp.enabled:
        dynamic_scale_instance = dynamic_scale.DynamicScale(
            init_scale=cfg.training.amp.loss_scale.init_scale,
            growth_interval=cfg.training.amp.loss_scale.growth_interval
        )
    
    state = train_state.TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=optimizer,
        dynamic_scale=dynamic_scale_instance
    )
    
    return state

def log_metrics(
    metrics: Dict[str, float],
    step: int,
    tb_writer: Optional[SummaryWriter] = None,
    wandb_run: Optional[wandb.Run] = None
) -> None:
    """Log metrics to configured monitoring tools."""
    # Log to console
    metric_str = " ".join(f"{k}: {v:.4f}" for k, v in metrics.items())
    logger.info(f"Step {step}: {metric_str}")
    
    # Log to TensorBoard
    if tb_writer is not None:
        for name, value in metrics.items():
            tb_writer.add_scalar(name, value, step)
    
    # Log to Weights & Biases
    if wandb_run is not None:
        wandb_run.log(metrics, step=step)

def save_checkpoint_with_metrics(
    state: train_state.TrainState,
    save_dir: str,
    step: int,
    metrics: Dict[str, float],
    cfg: DictConfig
) -> None:
    """Save checkpoint with metric tracking for best model selection."""
    state = unreplicate(state)
    
    # Save checkpoint
    checkpoints.save_checkpoint(
        ckpt_dir=save_dir,
        target=state,
        step=step,
        keep=cfg.training.checkpointing.keep_last_n,
        overwrite=True
    )
    
    # Track best model if configured
    if cfg.training.checkpointing.save_best:
        metric = metrics[cfg.training.checkpointing.metric]
        is_better = metric < state.best_metric if cfg.training.checkpointing.mode == "min" else metric > state.best_metric
        
        if is_better:
            state = state.replace(best_metric=metric)
            checkpoints.save_checkpoint(
                ckpt_dir=os.path.join(save_dir, "best"),
                target=state,
                step=step,
                overwrite=True
            )

def train_step(
    state: train_state.TrainState,
    batch: Dict[str, jnp.ndarray],
    cfg: DictConfig,
    rng: jax.random.PRNGKey
) -> Tuple[train_state.TrainState, Dict[str, float], jax.random.PRNGKey]:
    """Execute training step with gradient accumulation and mixed precision."""
    
    def loss_fn(params):
        outputs = state.apply_fn(
            {'params': params},
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            train=True,
            rngs={'dropout': rng}
        )
        
        # Calculate losses
        loss = outputs['loss']
        if 'aux_loss' in outputs:
            loss = loss + cfg.model.moe_config.load_balance_weight * outputs['aux_loss']
            
        metrics = {
            'loss': loss,
            'accuracy': outputs.get('accuracy', 0.0),
            'perplexity': jnp.exp(loss),
        }
        
        if 'aux_loss' in outputs:
            metrics['aux_loss'] = outputs['aux_loss']
            
        return loss, metrics
    
    # Handle mixed precision training
    if state.dynamic_scale:
        dynamic_scale, is_finite, aux = state.dynamic_scale.value_and_grad(
            loss_fn, has_aux=True)(state.params)
        
        # Unpack gradients and metrics
        grad, metrics = aux
        
        # Update dynamic scale
        dynamic_scale = dynamic_scale.update(is_finite)
        state = state.replace(dynamic_scale=dynamic_scale)
        
        # Clear gradients if not finite
        grad = jax.tree_map(
            lambda x: jnp.where(is_finite, x, jnp.zeros_like(x)),
            grad
        )
    else:
        (loss, metrics), grad = jax.value_and_grad(
            loss_fn, has_aux=True)(state.params)
    
    # Average gradients across devices
    grad = jax.lax.pmean(grad, axis_name='batch')
    metrics = jax.lax.pmean(metrics, axis_name='batch')
    
    # Update optimizer state
    state = state.apply_gradients(grads=grad)
    
    # Generate new RNG key
    rng, new_rng = jax.random.split(rng)
    
    return state, metrics, new_rng

@hydra.main(config_path="configs", config_name="default_config")
def main(cfg: DictConfig) -> None:
    """Main training function using Hydra for configuration."""
    # Setup logging and monitoring
    tb_writer, wandb_run = setup_monitoring(cfg)
    
    # Initialize random state
    rng = jax.random.PRNGKey(cfg.get('seed', 42))
    
    # Create student model
    student_model = VishwamAIModel(ModelConfig(**cfg.model))
    tokenizer = VishwamAITokenizer.from_pretrained(cfg.tokenizer_path)
    
    # Setup distillation if enabled
    distillation_trainer = None
    if cfg.get('distillation', {}).get('enabled', False):
        if not cfg.distillation.teacher_model.path:
            raise ValueError("Teacher model path must be provided for distillation")
        
        # Load teacher model
        teacher_config = ModelConfig(**cfg.model)  # Can be different config
        teacher_model = VishwamAIModel(teacher_config)
        teacher_state = checkpoints.restore_checkpoint(
            cfg.distillation.teacher_model.path,
            target=None
        )
        teacher_model.params = teacher_state['params']
        
        # Create distillation trainer
        distillation_trainer = DistillationTrainer(
            teacher_model=teacher_model,
            student_model=student_model,
            cfg=cfg
        )
    
    # Initialize auxiliary components
    error_correction = None
    tot = None
    
    if cfg.model.error_correction.enabled:
        error_correction = ModelIntegrator(cfg.model.error_correction)
    
    if cfg.model.tot_config.enabled:
        tot = TreeOfThoughts(cfg.model.tot_config)
    
    # Setup hyperparameter tuning if enabled
    if cfg.tuning.enabled:
        study = optuna.create_study(
            study_name=cfg.tuning.study_name,
            direction=cfg.tuning.direction,
            pruner=instantiate(cfg.tuning.pruner),
            sampler=instantiate(cfg.tuning.sampler)
        )
        
        def objective(trial):
            # Suggest hyperparameters
            for name, param in cfg.tuning.parameters.items():
                if param.type == "loguniform":
                    cfg[name] = trial.suggest_loguniform(name, param.low, param.high)
                elif param.type == "uniform":
                    cfg[name] = trial.suggest_uniform(name, param.low, param.high)
                elif param.type == "int":
                    cfg[name] = trial.suggest_int(name, param.low, param.high)
            
            # Create model instance for training
            model = VishwamAIModel(ModelConfig(**cfg.model))
            
            # Train with suggested parameters
            state = create_train_state_with_amp(model, cfg, rng)
            state = replicate(state)
            
            train_loader = create_train_dataloader(cfg)
            val_loader = create_val_dataloader(cfg)
            
            for step in range(cfg.training.max_steps):
                # Perform training step
                if distillation_trainer:
                    state, metrics, rng = distillation_trainer.train_step(
                        state,
                        next(train_loader),
                        step,
                        rng
                    )
                else:
                    state, metrics, rng = train_step(state, next(train_loader), cfg, rng)
                
                if step % cfg.training.eval_steps == 0:
                    val_metrics = evaluate(state, val_loader, cfg)
                    trial.report(val_metrics[cfg.tuning.metric], step)
                    
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            
            return val_metrics[cfg.tuning.metric]
        
        study.optimize(objective, n_trials=cfg.tuning.num_trials)
        best_params = study.best_params
        cfg.update(best_params)
    
    # Create model instance
    model = VishwamAIModel(ModelConfig(**cfg.model))
    
    # Create initial training state
    state = create_train_state_with_amp(model, cfg, rng)
    state = replicate(state)
    
    # Training loop
    train_loader = create_train_dataloader(cfg)
    val_loader = create_val_dataloader(cfg)
    
    for step in tqdm(range(cfg.training.max_steps)):
        state, metrics, rng = train_step(state, next(train_loader), cfg, rng)
        
        # Log metrics
        if step % cfg.training.logging_steps == 0:
            log_metrics(metrics, step, tb_writer, wandb_run)
        
        # Evaluate and save checkpoints
        if step % cfg.training.eval_steps == 0:
            val_metrics = evaluate(state, val_loader, cfg)
            log_metrics(val_metrics, step, tb_writer, wandb_run)
            
            if step % cfg.training.save_steps == 0:
                save_checkpoint_with_metrics(
                    state,
                    cfg.training.checkpointing.dir,
                    step,
                    val_metrics,
                    cfg
                )
    
    # Post-training quantization if enabled
    if (distillation_trainer and 
        cfg.distillation.quantization.enabled):
        state = distillation_trainer.quantize_model(
            state,
            val_loader,
            cfg.distillation.quantization.calibration_steps
        )
        
        # Save quantized model
        save_checkpoint_with_metrics(
            state,
            os.path.join(cfg.training.checkpointing.dir, "quantized"),
            cfg.training.max_steps,
            metrics,
            cfg
        )
    
    # Cleanup
    if tb_writer is not None:
        tb_writer.close()
    if wandb_run is not None:
        wandb_run.finish()

if __name__ == '__main__':
    main()
