#!/usr/bin/env python3
"""
TPU-Optimized Training Script for VishwamAI with resource-conscious optimization.

This script implements advanced techniques to maximize training efficiency with limited TPU resources:
1. Gradient accumulation for virtual batch sizes
2. Memory-efficient checkpointing
3. Dynamic batch size scaling
4. Mixed precision training with bfloat16
5. Smart sharding across TPU devices
6. Automatic memory profiling and optimization
"""

import os
import time
import logging
import argparse
import json
import gc
from typing import Dict, Any, Optional, Tuple, List
from functools import partial
from datetime import datetime
import psutil

import numpy as np
import jax
import jax.numpy as jnp
from jax.experimental import mesh_utils
from jax.sharding import Mesh, PartitionSpec
import optax
import flax
from flax import linen as nn
from flax.training import train_state, checkpoints
import orbax.checkpoint

from tqdm import tqdm
from omegaconf import OmegaConf, DictConfig

from vishwamai.model import VishwamAIModel, ModelConfig
from vishwamai.tokenizer import VishwamAITokenizer
from vishwamai.data_utils import create_train_dataloader, create_val_dataloader
from vishwamai.error_correction import ErrorCorrectionTrainer, create_error_corrected_train_step
from vishwamai.tot import TreeOfThoughts
from vishwamai.integration import ToTIntegrationLayer, ToTModelIntegrator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("tpu_training.log")
    ]
)
logger = logging.getLogger(__name__)

# Memory tracking utilities
def log_memory_usage():
    """Log memory usage on TPU and host."""
    try:
        process = psutil.Process(os.getpid())
        host_memory_gb = process.memory_info().rss / (1024 * 1024 * 1024)
        
        logger.info(f"Host memory usage: {host_memory_gb:.2f} GB")
        
        # Try to get TPU memory stats using multiple approaches
        tpu_memory_logged = False
        
        # Try PyTorch XLA approach first (for PyTorch+XLA environments)
        try:
            # Check if torch_xla is available before attempting to import metrics
            import importlib.util
            if importlib.util.find_spec("torch_xla") is not None:
                # Only import if the module exists
                import torch_xla.debug.metrics as torch_xla_metrics
                tpu_memory_usage = torch_xla_metrics.metrics_report()
                logger.info(f"TPU memory report (PyTorch XLA): {tpu_memory_usage}")
                tpu_memory_logged = True
            else:
                logger.debug("PyTorch XLA package not available")
        except ImportError:
            logger.debug("PyTorch XLA metrics import failed, will try JAX approach")
        except Exception as e:
            logger.debug(f"Error getting PyTorch XLA metrics: {e}")
            
        # Fallback to JAX-specific TPU memory tracking
        if not tpu_memory_logged:
            try:
                memory_stats = jax.peak_memory_stats()
                for d, stats in memory_stats.items():
                    logger.info(f"TPU {d} peak memory: {stats['peak_bytes'] / (1024**3):.2f} GB")
                tpu_memory_logged = True
            except Exception as e:
                logger.debug(f"Error getting JAX TPU metrics: {e}")
                
        # If neither approach worked, log a message
        if not tpu_memory_logged:
            logger.info("Could not retrieve TPU memory metrics - neither PyTorch XLA nor JAX metrics available")
            
    except Exception as e:
        logger.warning(f"Failed to log memory usage: {e}")

def setup_tpu_devices():
    """Set up TPU devices with optimal configuration."""
    try:
        # Check if TPU is available
        tpu_devices = jax.devices("tpu")
        num_devices = len(tpu_devices)
        
        if (num_devices == 0):
            logger.warning("No TPU devices found, falling back to CPU/GPU")
            return None, jax.devices()
        
        # Create a mesh for model parallelism
        device_mesh = mesh_utils.create_device_mesh((num_devices,))
        mesh = Mesh(device_mesh, axis_names=('data',))
        
        logger.info(f"TPU mesh created with {num_devices} devices")
        for i, device in enumerate(tpu_devices):
            logger.info(f"TPU {i}: {device.platform}:{device.id}")
        
        # Release memory we don't need
        gc.collect()
        
        return mesh, tpu_devices
    except Exception as e:
        logger.error(f"Error setting up TPU: {e}")
        return None, jax.devices()

class GradientAccumulator:
    """Efficient gradient accumulation for virtual batch sizes."""
    
    def __init__(self, steps: int = 1):
        self.steps = max(1, steps)
        self.count = 0
        self.grads = None
        
    def reset(self):
        """Reset accumulator."""
        self.count = 0
        self.grads = None
    
    @partial(jax.jit, static_argnums=(0,))
    def add_gradients(self, grads):
        """Add gradients to accumulator."""
        if self.grads is None:
            self.grads = jax.tree_map(lambda g: g.copy(), grads)
        else:
            self.grads = jax.tree_map(lambda g1, g2: g1 + g2, self.grads, grads)
        self.count += 1
    
    @partial(jax.jit, static_argnums=(0,))
    def get_accumulated_gradients(self):
        """Get scaled accumulated gradients."""
        if self.grads is None:
            return None
        # Scale gradients by the number of accumulation steps
        return jax.tree_map(lambda g: g / max(1, self.count), self.grads)
    
    def should_update(self):
        """Check if we should update weights."""
        return self.count >= self.steps

class TPUTrainer:
    """Resource-efficient TPU Trainer with advanced memory optimization."""
    
    def __init__(self, config: DictConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.train_state = None
        self.tpu_mesh = None
        self.grad_accumulator = GradientAccumulator(
            steps=config.training.gradient_accumulation_steps
        )
        
        # Advanced memory optimization parameters
        self.use_gradient_checkpointing = config.training.get('use_gradient_checkpointing', True)
        self.use_bfloat16 = config.training.get('use_bfloat16', True)
        self.use_dualpipe = config.training.get('use_dualpipe', True)
        self.max_allowed_batch_size = config.training.get('max_batch_size', 32)
        self.grad_clip = config.training.get('grad_clip', 1.0)
        
        # Tree of Thoughts and Error Correction
        self.use_error_correction = config.training.get('use_error_correction', False) 
        self.use_tot = config.training.get('use_tot', False)
        self.tot_search_strategy = config.training.get('tot_search_strategy', 'beam')
        self.tot_model = None
        self.error_trainer = None
        
        # Memory optimization for mixed precision
        self.param_dtype = jnp.bfloat16 if self.use_bfloat16 else jnp.float32
        self.compute_dtype = jnp.bfloat16 if self.use_bfloat16 else jnp.float32
        
        # Initialize TPU devices
        self.tpu_mesh, self.devices = setup_tpu_devices()
        self.num_devices = len(self.devices)
        
        # Orbax checkpointing for efficient saving/loading
        self.orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
        
        # Sharding for model parameters
        self.param_sharding = None
        
        # Dynamic batch sizing
        self.current_batch_size = config.training.get('initial_batch_size', 8)
        self.dynamic_batch_step = config.training.get('dynamic_batch_step', 0)
        self.batch_size_increase_step = config.training.get('batch_size_increase_step', 1000)
        self.batch_size_increase_factor = config.training.get('batch_size_increase_factor', 1.5)

    def _create_train_state(self, model_config: ModelConfig, rng: jax.random.PRNGKey):
        """Create training state with memory-efficient initialization."""
        logger.info("Creating training state...")
        
        # Create model with mixed precision option
        model_config.dtype = "bfloat16" if self.use_bfloat16 else "float32"
        model = VishwamAIModel(model_config)
        
        # Compute optimal parameter shapes for sharding
        sample_input = jnp.ones((1, 16), dtype=jnp.int32)
        params = model.init(rng, sample_input)["params"]
        
        # Create optimizer with gradient clipping and weight decay
        tx = optax.chain(
            optax.clip_by_global_norm(self.grad_clip),
            optax.adamw(
                learning_rate=self._create_learning_rate_schedule(),
                weight_decay=self.config.training.weight_decay,
                b1=self.config.training.adam_beta1,
                b2=self.config.training.adam_beta2
            )
        )
        
        # Create training state with additional metrics
        state = train_state.TrainState.create(
            apply_fn=model.apply,
            params=params,
            tx=tx
        )
        
        self.model = model
        return state
    
    def _create_learning_rate_schedule(self):
        """Create TPU-optimized learning rate schedule."""
        # Strong reference to values to avoid recompilation
        warmup_steps = self.config.training.warmup_steps
        total_steps = self.config.training.max_steps
        lr = self.config.training.learning_rate
        min_lr_ratio = self.config.training.get('min_lr_ratio', 0.1)
        
        def lr_schedule(step):
            """Cosine learning rate schedule with linear warmup."""
            warmup_factor = jnp.minimum(1.0, step / warmup_steps) if warmup_steps > 0 else 1.0
            decay_factor = 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(step, total_steps) / total_steps))
            decay_factor = min_lr_ratio + (1 - min_lr_ratio) * decay_factor
            return lr * warmup_factor * decay_factor
        
        return lr_schedule
    
    def _initialize_components(self):
        """Initialize training components with resource optimization."""
        try:
            # Load or create tokenizer
            tokenizer_path = self.config.tokenizer.get('path', None)
            if tokenizer_path and os.path.exists(tokenizer_path):
                logger.info(f"Loading tokenizer from {tokenizer_path}")
                self.tokenizer = VishwamAITokenizer.from_pretrained(tokenizer_path)
            else:
                logger.info("Creating new tokenizer")
                self.tokenizer = VishwamAITokenizer(
                    vocab_size=self.config.model.vocab_size
                )
            
            # Random seeds for reproducibility
            rng = jax.random.PRNGKey(self.config.training.seed)
            
            # Load model configuration
            model_config = ModelConfig(**vars(self.config.model))
            
            # Create training state
            self.train_state = self._create_train_state(model_config, rng)
            
            # Initialize Tree of Thoughts if enabled
            if self.use_tot:
                logger.info("Initializing Tree of Thoughts...")
                self.tot_model = TreeOfThoughts(
                    transformer=self.model,
                    tokenizer=self.tokenizer,
                    max_thoughts=self.config.tot.get('max_thoughts', 5),
                    max_depth=self.config.tot.get('max_depth', 3),
                    beam_width=self.config.tot.get('beam_width', 5),
                    use_tpu=True
                )
            
            # Initialize error correction if enabled
            if self.use_error_correction:
                logger.info("Initializing Error Correction...")
                self.error_trainer = ErrorCorrectionTrainer(
                    config=self.config,
                    transformer=self.model,
                    tokenizer=self.tokenizer,
                    use_tot=self.use_tot,
                    use_mod=True,
                    use_bfloat16=self.use_bfloat16,
                    use_dualpipe=self.use_dualpipe
                )
                
                # Initialize error correction parameters
                rng, ec_rng = jax.random.split(rng)
                hidden_size = self.config.model.hidden_size
                sample_features = jnp.ones((1, 1, hidden_size), dtype=jnp.bfloat16 if self.use_bfloat16 else jnp.float32)
                self.error_trainer.init_params(ec_rng, sample_features)
            
            # Adjust batch size based on available TPU memory
            self._optimize_batch_size()
            
            logger.info("Initialization complete")
            
        except Exception as e:
            logger.error(f"Error during initialization: {e}")
            raise
    
    @partial(jax.jit, static_argnums=(0,))
    def _train_step(self, state, batch, rng):
        """JIT-compiled memory-efficient training step."""
        def loss_fn(params):
            outputs = self.model.apply(
                {'params': params},
                batch['input_ids'],
                deterministic=False,
                rngs={'dropout': rng},
                output_hidden_states=True
            )
            
            logits = outputs['logits']
            
            # Shift logits and labels for next token prediction
            logits = logits[:, :-1]
            labels = batch['input_ids'][:, 1:]
            
            # Create mask to ignore padding tokens
            mask = jnp.where(labels > 0, 1.0, 0.0)
            
            # Calculate cross-entropy loss
            loss = optax.softmax_cross_entropy_with_integer_labels(
                logits, labels
            )
            loss = (loss * mask).sum() / mask.sum()
            
            # Calculate metrics
            metrics = {
                'loss': loss,
                'perplexity': jnp.exp(jnp.minimum(loss, 100.0))
            }
            
            return loss, (metrics, outputs)
        
        # Get gradients with memory-efficient gradient checkpointing
        (loss, (metrics, outputs)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # No weight updates here if using gradient accumulation
        return grads, metrics, outputs
    
    def _error_corrected_train_step(self, state, batch, rng):
        """Memory-efficient train step with error correction."""
        # Use regular train step to get gradients and outputs
        grads, metrics, outputs = self._train_step(state, batch, rng)
        
        # Apply error correction
        if self.use_error_correction:
            rng, ec_rng = jax.random.split(rng)
            ec_outputs = self.error_trainer.apply_error_correction(
                logits=outputs['logits'],
                features=outputs['hidden_states'],
                labels=batch['input_ids'],
                training=True,
                rng_key=ec_rng
            )
            
            # Enhance metrics with error correction info
            metrics['error_correction_rate'] = jnp.mean(ec_outputs['correction_mask'].astype(float))
            metrics['detection_loss'] = ec_outputs.get('detection_loss', 0.0)
            
            # Scale gradients to include error correction 
            correction_weight = self.config.training.get('error_correction_weight', 0.5)
            if ec_outputs.get('detection_loss') is not None:
                # Scale added loss with a weight factor
                correction_grads = jax.grad(
                    lambda p: correction_weight * ec_outputs['detection_loss']
                )(state.params)
                
                # Add to original gradients
                grads = jax.tree_map(
                    lambda g1, g2: g1 + g2,
                    grads,
                    correction_grads
                )
        
        return grads, metrics, outputs
    
    def _update_weights(self, state, accumulated_grads):
        """Update weights using accumulated gradients."""
        # Apply gradients
        return state.apply_gradients(grads=accumulated_grads)
    
    def _optimize_batch_size(self):
        """Dynamically optimize batch size based on TPU memory."""
        try:
            # Start with current batch size
            batch_size = self.current_batch_size
            max_batch_size = self.max_allowed_batch_size
            
            # Try increasing batch size
            test_sizes = []
            for size_factor in [1.0, 1.2, 1.5, 2.0]:
                test_size = min(
                    int(batch_size * size_factor),
                    max_batch_size
                )
                if test_size not in test_sizes:
                    test_sizes.append(test_size)
            
            # Find largest working batch size
            optimal_size = batch_size
            for test_size in sorted(test_sizes):
                try:
                    # Create a small sample batch
                    sample_batch = {
                        'input_ids': jnp.ones((test_size, 64), dtype=jnp.int32)
                    }
                    
                    # Try running a single step with this batch size
                    rng = jax.random.PRNGKey(0)
                    _ = self._train_step(self.train_state, sample_batch, rng)
                    
                    # If successful, update optimal size
                    optimal_size = test_size
                    logger.info(f"Successfully tested batch size: {test_size}")
                except (RuntimeError, jax.errors.OutOfMemoryError) as e:
                    logger.info(f"Batch size {test_size} failed: {e}")
                    break
            
            # Update current batch size
            self.current_batch_size = optimal_size
            logger.info(f"Optimized batch size: {self.current_batch_size}")
            
        except Exception as e:
            logger.error(f"Error during batch size optimization: {e}")
    
    def _save_checkpoint(self, step: int):
        """Memory-efficient checkpoint saving."""
        try:
            checkpoint_dir = self.config.training.checkpoint_dir
            os.makedirs(checkpoint_dir, exist_ok=True)
            
            # Use Orbax to save checkpoint efficiently
            ckpt_path = os.path.join(checkpoint_dir, f"step_{step}")
            
            # Convert params to CPU before saving to free GPU memory
            cpu_state = jax.device_get(self.train_state)
            
            # Save efficiently with Orbax
            save_args = orbax.checkpoint.SaveArgs(
                save_paths=ckpt_path
            )
            self.orbax_checkpointer.save(ckpt_path, cpu_state, save_args=save_args)
            
            # Save model configuration alongside checkpoint
            config_path = os.path.join(checkpoint_dir, "model_config.json")
            with open(config_path, 'w') as f:
                json.dump(vars(self.config.model), f, indent=2)
            
            logger.info(f"Checkpoint saved to {ckpt_path}")
            
            # Force garbage collection to free memory
            del cpu_state
            gc.collect()
            
        except Exception as e:
            logger.error(f"Error saving checkpoint: {e}")
    
    def _load_checkpoint(self, path: str):
        """Memory-efficient checkpoint loading."""
        try:
            # Use Orbax to load checkpoint efficiently
            restored_state = self.orbax_checkpointer.restore(path, item=self.train_state)
            self.train_state = restored_state
            logger.info(f"Checkpoint loaded from {path}")
            return True
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")
            return False
    
    def train(self):
        """Run training loop with TPU optimization."""
        try:
            logger.info("Starting training...")
            self._initialize_components()
            
            # Create data loaders
            train_loader = create_train_dataloader(self.config)
            val_loader = create_val_dataloader(self.config) if self.config.training.get('do_eval', True) else None
            
            # Resume from checkpoint if specified
            resume_from = self.config.training.get('resume_from', None)
            if resume_from:
                logger.info(f"Resuming from checkpoint: {resume_from}")
                success = self._load_checkpoint(resume_from)
                if not success:
                    logger.warning("Failed to load checkpoint, starting from scratch")
            
            # Training loop
            steps_per_epoch = self.config.training.steps_per_epoch
            num_epochs = self.config.training.epochs
            max_steps = self.config.training.max_steps
            
            # Initial logging
            log_memory_usage()
            
            # Main training loop with progress bar
            log_every = self.config.training.get('log_every', 10)
            eval_every = self.config.training.get('eval_every', 500)
            save_every = self.config.training.get('save_every', 1000)
            
            global_step = int(self.train_state.step)
            epoch = global_step // steps_per_epoch
            
            rng = jax.random.PRNGKey(self.config.training.seed + global_step)
            
            all_metrics = []
            start_time = time.time()
            
            with tqdm(total=max_steps, initial=global_step, dynamic_ncols=True) as pbar:
                while global_step < max_steps and epoch < num_epochs:
                    epoch_metrics = []
                    
                    # Reset gradient accumulator
                    self.grad_accumulator.reset()
                    
                    for step in range(min(steps_per_epoch, max_steps - global_step)):
                        # Get batch with current batch size
                        batch = next(train_loader)
                        
                        # Split RNG for different components
                        rng, step_rng, ec_rng = jax.random.split(rng, 3)
                        
                        # Forward & backward pass
                        if self.use_error_correction:
                            grads, metrics, _ = self._error_corrected_train_step(
                                self.train_state, batch, step_rng
                            )
                        else:
                            grads, metrics, _ = self._train_step(
                                self.train_state, batch, step_rng
                            )
                        
                        # Accumulate gradients
                        self.grad_accumulator.add_gradients(grads)
                        epoch_metrics.append(metrics)
                        
                        # Update weights if we've accumulated enough gradients
                        if self.grad_accumulator.should_update():
                            accumulated_grads = self.grad_accumulator.get_accumulated_gradients()
                            self.train_state = self._update_weights(self.train_state, accumulated_grads)
                            self.grad_accumulator.reset()
                        
                        # Increase global step only after weight update
                        global_step = int(self.train_state.step)
                        
                        # Log metrics
                        if global_step % log_every == 0:
                            # Average metrics from accumulated steps
                            avg_metrics = {
                                k: np.mean([m[k] for m in epoch_metrics[-log_every:]])
                                for k in epoch_metrics[-1].keys()
                            }
                            
                            # Calculate throughput
                            elapsed = time.time() - start_time
                            throughput = log_every * self.current_batch_size / elapsed
                            
                            # Log to console and update progress bar
                            log_str = f"Step {global_step}: loss={avg_metrics['loss']:.4f}, " \
                                      f"ppl={avg_metrics['perplexity']:.4f}, " \
                                      f"bs={self.current_batch_size}, " \
                                      f"throughput={throughput:.1f} samples/sec"
                                      
                            if 'error_correction_rate' in avg_metrics:
                                log_str += f", err_corr={avg_metrics['error_correction_rate']:.2f}"
                                
                            pbar.set_postfix_str(log_str)
                            pbar.update(log_every)
                            
                            # Log memory usage periodically
                            if global_step % (log_every * 10) == 0:
                                log_memory_usage()
                                
                            # Reset timer
                            start_time = time.time()
                        
                        # Dynamic batch size adjustment
                        if self.config.training.get('dynamic_batch_size', False) and \
                           global_step % self.batch_size_increase_step == 0 and \
                           global_step > 0:
                            prev_bs = self.current_batch_size
                            self._optimize_batch_size()
                            if (self.current_batch_size != prev_bs):
                                logger.info(f"Batch size adjusted: {prev_bs} → {self.current_batch_size}")
                                
                        # Run evaluation
                        if val_loader and global_step % eval_every == 0 and global_step > 0:
                            eval_metrics = self._run_evaluation(val_loader)
                            logger.info(f"Eval step {global_step}: {eval_metrics}")
                                
                        # Save checkpoint
                        if global_step % save_every == 0 and global_step > 0:
                            self._save_checkpoint(global_step)
                    
                    # End of epoch
                    epoch += 1
                    logger.info(f"Finished epoch {epoch}")
                    all_metrics.extend(epoch_metrics)
                    
                    # Save checkpoint at end of epoch
                    self._save_checkpoint(global_step)
            
            # Final checkpoint
            self._save_checkpoint(global_step)
            logger.info("Training complete!")
            
        except Exception as e:
            logger.error(f"Error during training: {e}")
            # Try to save checkpoint even if training fails
            try:
                if hasattr(self, 'train_state') and self.train_state:
                    self._save_checkpoint(int(self.train_state.step))
                    logger.info("Saved checkpoint after error")
            except:
                pass
            raise
    
    def _run_evaluation(self, val_loader, num_batches: int = 10):
        """Run memory-efficient evaluation."""
        try:
            metrics_list = []
            
            for _ in range(num_batches):
                try:
                    batch = next(val_loader)
                    
                    # Run forward pass in eval mode
                    outputs = self.model.apply(
                        {'params': self.train_state.params},
                        batch['input_ids'],
                        deterministic=True
                    )
                    
                    logits = outputs['logits']
                    
                    # Calculate loss
                    shifted_logits = logits[:, :-1]
                    labels = batch['input_ids'][:, 1:]
                    
                    mask = jnp.where(labels > 0, 1.0, 0.0)
                    loss = optax.softmax_cross_entropy_with_integer_labels(
                        shifted_logits, labels
                    )
                    loss = (loss * mask).sum() / mask.sum()
                    
                    # Add metrics
                    batch_metrics = {
                        'eval_loss': float(loss),
                        'eval_perplexity': float(jnp.exp(jnp.minimum(loss, 100.0)))
                    }
                    
                    metrics_list.append(batch_metrics)
                
                except Exception as e:
                    logger.error(f"Error during evaluation batch: {e}")
                    continue
            
            # Average metrics
            avg_metrics = {}
            if metrics_list:
                for k in metrics_list[0].keys():
                    avg_metrics[k] = float(np.mean([m[k] for m in metrics_list]))
            
            return avg_metrics
            
        except Exception as e:
            logger.error(f"Error during evaluation: {e}")
            return {'eval_error': str(e)}

def main():
    parser = argparse.ArgumentParser(description="TPU-Optimized Training for VishwamAI")
    parser.add_argument('--config', type=str, required=True, help="Path to config file")
    parser.add_argument('--wandb', action='store_true', help="Enable W&B logging")
    parser.add_argument('--profile', action='store_true', help="Enable JAX profiling")
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    # Enable JAX profiling if requested
    if args.profile:
        try:
            from jax import profiler
            profile_dir = "./jax_profile"
            os.makedirs(profile_dir, exist_ok=True)
            profiler.start_trace(profile_dir)
            logger.info(f"JAX profiling enabled, results will be saved to {profile_dir}")
        except:
            logger.warning("Failed to enable JAX profiling")
    
    # Enable WandB logging if requested
    if args.wandb:
        try:
            import wandb
            wandb.init(
                project=config.get('wandb_project', "vishwamai"),
                name=config.get('wandb_run_name', f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"),
                config=OmegaConf.to_container(config)
            )
            logger.info("W&B logging enabled")
        except:
            logger.warning("Failed to enable W&B logging")
    
    # Create and run trainer
    trainer = TPUTrainer(config)
    trainer.train()
    
    # Stop JAX profiling if enabled
    if args.profile:
        try:
            profiler.stop_trace()
            logger.info("JAX profiling complete")
        except:
            pass

if __name__ == "__main__":
    main()