import jax
import jax.numpy as jnp
import flax
import optax
from flax.training import train_state
from typing import Dict, List, Tuple, Any, Optional, Union
import numpy as np
from tqdm import tqdm
import os
import math
from omegaconf import OmegaConf
import safetensors.flax as stf

from vishwamai.training import TrainingState, cross_entropy_loss, create_optimizer

class VishwamaiGuruKnowledge:
    """Knowledge distillation configuration and utilities."""
    
    def __init__(self, config):
        """Initialize guru knowledge with config."""
        self.config = config
        self.temperature = config.distillation.temperature
        self.alpha = config.distillation.alpha  # Weight for distillation loss
        
        # Feature matching configuration
        self.feature_distill = config.distillation.get('feature_distill', False)
        self.hidden_distill = config.distillation.get('hidden_distill', False)
        self.attention_distill = config.distillation.get('attention_distill', False)
        
        # Weights for different feature matching components
        self.hidden_alpha = config.distillation.get('hidden_alpha', 0.0)
        self.attention_alpha = config.distillation.get('attention_alpha', 0.0)
        
        # Layer mapping between teacher and student
        self.layer_mapping = self._create_layer_mapping(
            teacher_layers=config.distillation.teacher_model.config.num_layers,
            student_layers=config.distillation.student_model.config.num_layers
        )
    
    def _create_layer_mapping(self, teacher_layers: int, student_layers: int) -> Dict[int, int]:
        """Create mapping between teacher and student layers."""
        # For evenly spaced mappings
        if student_layers < teacher_layers:
            indices = np.linspace(0, teacher_layers - 1, student_layers, dtype=int)
            return {s: t for s, t in enumerate(indices)}
        # For one-to-one mapping when sizes match
        elif student_layers == teacher_layers:
            return {i: i for i in range(student_layers)}
        # For over-parameterized student (less common)
        else:
            indices = np.linspace(0, student_layers - 1, teacher_layers, dtype=int)
            mapping = {}
            for t, s in enumerate(indices):
                if s not in mapping:
                    mapping[s] = []
                mapping[s].append(t)
            # Convert to a one-to-many mapping
            return {s: t[0] if len(t) == 1 else t for s, t in mapping.items()}
    
    def kl_divergence(self, student_logits: jnp.ndarray, teacher_logits: jnp.ndarray) -> jnp.ndarray:
        """Calculate KL divergence loss for knowledge distillation."""
        # Apply temperature scaling
        student_logits_t = student_logits / self.temperature
        teacher_logits_t = teacher_logits / self.temperature
        
        # Convert logits to probabilities
        student_probs = jax.nn.softmax(student_logits_t, axis=-1)
        teacher_probs = jax.nn.softmax(teacher_logits_t, axis=-1)
        
        # Calculate KL divergence
        loss = jnp.sum(teacher_probs * (jnp.log(teacher_probs + 1e-10) - jnp.log(student_probs + 1e-10)), axis=-1)
        
        # Apply temperature^2 scaling as per the distillation paper
        loss = loss * (self.temperature ** 2)
        
        return loss
    
    def hidden_mse_loss(self, student_hidden: jnp.ndarray, teacher_hidden: jnp.ndarray) -> jnp.ndarray:
        """Calculate MSE loss for hidden state matching."""
        # Optionally apply dimensionality reduction if sizes don't match
        if student_hidden.shape[-1] != teacher_hidden.shape[-1]:
            # Project to smaller dimension
            dim = min(student_hidden.shape[-1], teacher_hidden.shape[-1])
            if student_hidden.shape[-1] > dim:
                student_hidden = student_hidden[..., :dim]
            if teacher_hidden.shape[-1] > dim:
                teacher_hidden = teacher_hidden[..., :dim]
        
        # Calculate MSE loss
        loss = jnp.mean(jnp.square(student_hidden - teacher_hidden))
        return loss
    
    def attention_matching_loss(self, student_attentions: jnp.ndarray, teacher_attentions: jnp.ndarray) -> jnp.ndarray:
        """Calculate attention map matching loss."""
        # Handle different attention head counts
        if student_attentions.shape[1] != teacher_attentions.shape[1]:
            # Average teacher attention heads if there are more of them
            if teacher_attentions.shape[1] > student_attentions.shape[1]:
                # Reshape and average groups of teacher attention heads
                num_student_heads = student_attentions.shape[1]
                num_teacher_heads = teacher_attentions.shape[1]
                
                # Ensure the teacher heads can be evenly grouped
                if num_teacher_heads % num_student_heads == 0:
                    # Reshape to batch, groups, heads_per_group, seq, seq
                    group_size = num_teacher_heads // num_student_heads
                    grouped_shape = (teacher_attentions.shape[0], num_student_heads, group_size) + teacher_attentions.shape[2:]
                    grouped_attentions = teacher_attentions.reshape(grouped_shape)
                    
                    # Average the heads in each group
                    teacher_attentions = jnp.mean(grouped_attentions, axis=2)
                else:
                    # Just select a subset of heads
                    indices = jnp.linspace(0, num_teacher_heads - 1, num_student_heads, dtype=int)
                    teacher_attentions = teacher_attentions[:, indices]
        
        # L1 distance is often better for attention matching
        loss = jnp.mean(jnp.abs(student_attentions - teacher_attentions))
        return loss


class VishwamaiShaalaTrainer:
    """Trainer for knowledge distillation."""
    
    def __init__(self, teacher_model, student_model, cfg):
        """Initialize trainer with teacher and student models."""
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.cfg = cfg
        
        # Create guru knowledge for distillation guidance
        self.guru = VishwamaiGuruKnowledge(cfg)
        
        # Tracking metrics
        self.step = 0
        self.best_metrics = {
            'distill_loss': float('inf'),
            'val_loss': float('inf')
        }
        
    def create_train_state(self, rng):
        """Initialize student model training state."""
        # Initialize student parameters if needed
        if not hasattr(self.student_model, 'params'):
            # Get sample input shape from config
            seq_len = self.cfg.model.max_seq_length
            batch_size = self.cfg.training.batch_size
            
            # Create sample input for initialization
            sample_input = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
            sample_mask = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
            
            # Initialize student model
            params_rng, dropout_rng = jax.random.split(rng)
            variables = self.student_model.init(
                {'params': params_rng, 'dropout': dropout_rng},
                input_ids=sample_input,
                attention_mask=sample_mask,
                deterministic=False
            )
            self.student_model.params = variables['params']
        
        # Create optimizer for student model
        tx, _ = create_optimizer(self.cfg)
        
        # Create training state
        state = TrainingState.create(
            apply_fn=self.student_model.__call__,
            params=self.student_model.params,
            tx=tx,
            ema_params=None,
            step=0,
            best_metrics=self.best_metrics
        )
        
        return state
    
    def train_step(self, state, batch, step, rng):
        """Perform a distillation training step."""
        dropout_rng = jax.random.fold_in(rng, step)
        
        # Get teacher model predictions (without gradients)
        teacher_outputs = self.teacher_model(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            deterministic=True,
            output_attentions=self.guru.attention_distill,
            output_hidden_states=self.guru.hidden_distill
        )
        
        teacher_logits = teacher_outputs[0]
        
        # Define the loss function for distillation
        def loss_fn(params):
            # Get student outputs with all the required components
            student_outputs = self.student_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                deterministic=False,
                output_attentions=self.guru.attention_distill,
                output_hidden_states=self.guru.hidden_distill,
                rngs={'dropout': dropout_rng}
            )
            
            student_logits = student_outputs[0]
            
            # Compute standard cross-entropy loss with shifted labels
            shift_logits = student_logits[:, :-1, :]
            shift_labels = batch['labels'][:, 1:]
            ce_loss = cross_entropy_loss(shift_logits, shift_labels)
            
            # Compute KL divergence for distillation
            kd_logits_s = student_logits[:, :-1, :]
            kd_logits_t = teacher_logits[:, :-1, :]
            kd_loss = jnp.mean(self.guru.kl_divergence(kd_logits_s, kd_logits_t))
            
            # Initialize feature matching losses
            hidden_loss = 0.0
            attn_loss = 0.0
            
            # Feature matching losses if enabled
            if self.guru.hidden_distill and len(student_outputs) > 1:
                student_hidden = student_outputs[1]
                teacher_hidden = teacher_outputs[1]
                
                # Match selected layers according to mapping
                for student_idx, teacher_idx in self.guru.layer_mapping.items():
                    if student_idx < len(student_hidden) and teacher_idx < len(teacher_hidden):
                        hidden_loss += self.guru.hidden_mse_loss(
                            student_hidden[student_idx], teacher_hidden[teacher_idx]
                        )
                
                # Average the loss over all mapped layers
                hidden_loss = hidden_loss / len(self.guru.layer_mapping)
            
            # Attention matching if enabled
            if self.guru.attention_distill and len(student_outputs) > 2:
                student_attn = student_outputs[2]
                teacher_attn = teacher_outputs[2]
                
                # Match selected layers according to mapping
                for student_idx, teacher_idx in self.guru.layer_mapping.items():
                    if student_idx < len(student_attn) and teacher_idx < len(teacher_attn):
                        attn_loss += self.guru.attention_matching_loss(
                            student_attn[student_idx], teacher_attn[teacher_idx]
                        )
                
                # Average the loss over all mapped layers
                attn_loss = attn_loss / len(self.guru.layer_mapping)
            
            # Combine all loss components
            total_loss = (
                (1 - self.guru.alpha) * ce_loss +
                self.guru.alpha * kd_loss +
                self.guru.hidden_alpha * hidden_loss +
                self.guru.attention_alpha * attn_loss
            )
            
            # Pack all metrics for logging
            metrics = {
                'ce_loss': ce_loss,
                'kd_loss': kd_loss,
                'hidden_loss': hidden_loss,
                'attention_loss': attn_loss,
                'total_loss': total_loss,
            }
            
            return total_loss, metrics
        
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (_, metrics), grads = grad_fn(state.params)
        
        # Update student parameters
        state = state.apply_gradients(grads=grads)
        
        # Update step count
        self.step = step
        
        # Get new RNG key for next step
        new_rng = jax.random.fold_in(rng, step + 1)
        
        return state, metrics, new_rng
    
    def evaluate(self, state, val_loader, teacher_model, guru):
        """Evaluate distilled model on validation data."""
        metrics_list = []
        
        # Number of batches to evaluate on
        num_eval_batches = min(100, self.cfg.training.get('eval_batches', 100))
        
        for i in range(num_eval_batches):
            try:
                batch = next(val_loader)
            except StopIteration:
                break
            
            # Get teacher predictions
            teacher_outputs = teacher_model(
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                deterministic=True,
                output_attentions=guru.attention_distill,
                output_hidden_states=guru.hidden_distill
            )
            
            teacher_logits = teacher_outputs[0]
            
            # Get student predictions
            student_outputs = jax.jit(state.apply_fn)(
                {'params': state.params},
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                deterministic=True,
                output_attentions=guru.attention_distill,
                output_hidden_states=guru.hidden_distill
            )
            
            student_logits = student_outputs[0]
            
            # Calculate metrics
            shift_logits_s = student_logits[:, :-1, :]
            shift_logits_t = teacher_logits[:, :-1, :]
            shift_labels = batch['labels'][:, 1:]
            
            # Cross entropy loss
            ce_loss = cross_entropy_loss(shift_logits_s, shift_labels)
            
            # KL divergence
            kd_loss = jnp.mean(guru.kl_divergence(shift_logits_s, shift_logits_t))
            
            # Calculate accuracy
            predictions = jnp.argmax(shift_logits_s, axis=-1)
            teacher_preds = jnp.argmax(shift_logits_t, axis=-1)
            
            # Mask padding tokens
            mask = (shift_labels != 0).astype(jnp.float32)
            
            # Student accuracy
            correct = (predictions == shift_labels).astype(jnp.float32) * mask
            accuracy = jnp.sum(correct) / (jnp.sum(mask) + 1e-8)
            
            # Teacher-student agreement
            agreement = (predictions == teacher_preds).astype(jnp.float32) * mask
            agreement_rate = jnp.sum(agreement) / (jnp.sum(mask) + 1e-8)
            
            # Store metrics
            batch_metrics = {
                'val_ce_loss': float(ce_loss),
                'val_kd_loss': float(kd_loss),
                'val_accuracy': float(accuracy),
                'val_agreement': float(agreement_rate),
                'val_perplexity': float(jnp.exp(ce_loss))
            }
            
            metrics_list.append(batch_metrics)
        
        # Average metrics across batches
        avg_metrics = {
            key: jnp.mean([batch[key] for batch in metrics_list])
            for key in metrics_list[0].keys()
        }
        
        # Update best metrics if improved
        if avg_metrics['val_ce_loss'] < self.best_metrics.get('val_loss', float('inf')):
            self.best_metrics['val_loss'] = float(avg_metrics['val_ce_loss'])
            self.best_metrics['val_accuracy'] = float(avg_metrics['val_accuracy'])
            self.best_metrics['is_best'] = True
        else:
            self.best_metrics['is_best'] = False
        
        return avg_metrics

    def save_checkpoint(self, state, path, guru=None, metadata=None):
        """Save student model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Prepare state dict with additional metadata
        checkpoint_dict = {
            'params': state.params,
            'step': self.step,
            'optimizer_state': state.opt_state,
            'metrics': self.best_metrics,
        }
        
        if guru is not None:
            checkpoint_dict['guru'] = {
                'temperature': guru.temperature,
                'alpha': guru.alpha,
                'hidden_alpha': guru.hidden_alpha,
                'attention_alpha': guru.attention_alpha
            }
        
        if metadata is not None:
            checkpoint_dict['metadata'] = metadata
        
        # Save checkpoint
        with open(f"{path}/checkpoint.msgpack", "wb") as f:
            f.write(flax.serialization.msgpack_serialize(checkpoint_dict))
        
        # Also save in safetensors format for better compatibility
        try:
            stf.save_file(
                {'params': state.params},
                f"{path}/model.safetensors"
            )
        except Exception as e:
            print(f"Warning: Could not save in safetensors format: {e}")
        
        print(f"Checkpoint saved at {path}")
        return path

    def load_checkpoint(self, state, path):
        """Load student model checkpoint."""
        try:
            with open(f"{path}/checkpoint.msgpack", "rb") as f:
                checkpoint_dict = flax.serialization.msgpack_restore(f.read())
            
            # Restore parameters
            state = state.replace(
                params=checkpoint_dict['params'],
                opt_state=checkpoint_dict.get('optimizer_state', state.opt_state),
                step=checkpoint_dict.get('step', 0)
            )
            
            # Restore metrics
            self.best_metrics = checkpoint_dict.get('metrics', self.best_metrics)
            self.step = checkpoint_dict.get('step', 0)
            
            # Restore guru if present
            if 'guru' in checkpoint_dict:
                guru_config = checkpoint_dict['guru']
                self.guru.temperature = guru_config.get('temperature', self.guru.temperature)
                self.guru.alpha = guru_config.get('alpha', self.guru.alpha)
                self.guru.hidden_alpha = guru_config.get('hidden_alpha', self.guru.hidden_alpha)
                self.guru.attention_alpha = guru_config.get('attention_alpha', self.guru.attention_alpha)
            
            print(f"Checkpoint loaded from {path}")
            return state, checkpoint_dict.get('metadata', {})
            
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            return state, {}

    def quantize_model(self, state, val_loader, num_calibration_steps=100):
        """Quantize the distilled model for deployment."""
        print("Quantizing model...")
        
        # This is a placeholder for actual quantization logic
        # In a full implementation, you would:
        # 1. Collect activation statistics across calibration data
        # 2. Determine optimal quantization parameters
        # 3. Apply quantization to weights and activations
        
        # Calibration loop to collect statistics
        for i in range(num_calibration_steps):
            try:
                batch = next(val_loader)
                # Run inference to collect activation statistics
                _ = jax.jit(state.apply_fn)(
                    {'params': state.params},
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    deterministic=True
                )
            except StopIteration:
                break
        
        print("Quantization complete")
        return state
