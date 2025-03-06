"""Qwen-specific distillation implementation."""
import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Optional

from .distillation import VishwamaiShaalaTrainer
from .layer_mapping import create_feature_mapping_config
from .loss_functions import compute_distillation_loss

class QwenDistillationTrainer(VishwamaiShaalaTrainer):
    """Specialized trainer for Qwen model distillation."""

    def __init__(self, teacher_model, student_model, cfg):
        super().__init__(teacher_model, student_model, cfg)
        # Create feature mapping configuration
        self.feature_mapping = create_feature_mapping_config(
            student_config=self.cfg.distillation.student_model.config,
            teacher_config=self.cfg.distillation.teacher_model.config
        )
        
        # Initialize any projection layers needed
        if self.feature_mapping['hidden_mapping']['needs_projection']:
            self._initialize_projection_layers()
    
    def _initialize_projection_layers(self):
        """Initialize projection layers for dimension matching if needed."""
        student_dim = self.feature_mapping['hidden_mapping']['student_dim']
        teacher_dim = self.feature_mapping['hidden_mapping']['teacher_dim']
        
        # Create projection matrix
        self.hidden_projection = jax.random.normal(
            jax.random.PRNGKey(42),
            (student_dim, teacher_dim)
        )
    
    @partial(jax.jit, static_argnums=(0,))
    def _project_hidden_states(self, hidden_states: jnp.ndarray) -> jnp.ndarray:
        """Project hidden states to match teacher dimensions."""
        if not self.feature_mapping['hidden_mapping']['needs_projection']:
            return hidden_states
        return jnp.matmul(hidden_states, self.hidden_projection)

    @partial(jax.jit, static_argnums=(0,))
    def _extract_layer_features(self, outputs: Dict[str, Any], is_teacher: bool = False) -> Dict[str, jnp.ndarray]:
        """Extract and align features from model outputs."""
        features = {}
        prefix = 'teacher_' if is_teacher else 'student_'
        
        # Extract hidden states
        if 'hidden_states' in outputs:
            for idx, hidden in enumerate(outputs['hidden_states']):
                layer_key = f'{prefix}layer_{idx}'
                features[layer_key] = hidden

        # Extract attention patterns
        if 'attentions' in outputs:
            for idx, attn in enumerate(outputs['attentions']):
                attn_key = f'{prefix}attention_{idx}'
                features[attn_key] = attn
        
        return features

    def _process_model_outputs(self, student_outputs: Dict[str, Any], teacher_outputs: Dict[str, Any]) -> Dict[str, Any]:
        """Process and align model outputs for distillation."""
        # Extract features
        student_features = self._extract_layer_features(student_outputs, is_teacher=False)
        teacher_features = self._extract_layer_features(teacher_outputs, is_teacher=True)
        
        # Project student features if needed
        if self.feature_mapping['hidden_mapping']['needs_projection']:
            for key in student_features:
                if 'layer_' in key:
                    student_features[key] = self._project_hidden_states(student_features[key])
        
        # Combine all outputs
        processed_outputs = {
            'student_logits': student_outputs['logits'],
            'teacher_logits': teacher_outputs['logits'],
            **student_features,
            **teacher_features
        }
        
        return processed_outputs

    @partial(jax.pmap, axis_name='batch', static_broadcasted_argnums=(0,))
    def _pmapped_train_step(self, state, batch, rng):
        """TPU-distributed training step with Qwen-specific optimizations."""
        rng, dropout_rng = jax.random.split(rng)
        
        def loss_fn(params):
            # Forward passes
            student_outputs = self.student_model.apply(
                {'params': params},
                batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                output_hidden_states=True,
                output_attentions=True,
                deterministic=False,
                rngs={'dropout': dropout_rng}
            )
            
            teacher_outputs = jax.lax.stop_gradient(
                self.teacher_model(
                    batch['input_ids'],
                    attention_mask=batch.get('attention_mask'),
                    output_hidden_states=True,
                    output_attentions=True,
                    deterministic=True
                )
            )
            
            # Process outputs for distillation
            processed_outputs = self._process_model_outputs(student_outputs, teacher_outputs)
            
            # Compute distillation loss
            loss, metrics = compute_distillation_loss(
                student_outputs=processed_outputs,
                teacher_outputs=processed_outputs,
                layer_mapping=self.feature_mapping['layer_mapping'],
                temperature=self.cfg.distillation_params.temperature,
                alpha_ce=self.cfg.distillation_params.alpha_ce,
                alpha_kd=self.cfg.distillation_params.alpha_kd,
                alpha_features=self.cfg.distillation_params.feature_loss_weight,
                alpha_attention=self.cfg.distillation_params.attention_loss_weight
            )
            
            return loss, (metrics, student_outputs)
        
        # Compute gradients
        (loss, (metrics, outputs)), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
        
        # All-reduce gradients across devices
        grads = jax.lax.pmean(grads, axis_name='batch')
        
        # Update state
        state = state.apply_gradients(grads=grads)
        
        return {'loss': loss, 'metrics': metrics, **outputs}, state

    def train_step(self, state, batch, rng):
        """Execute training step with automatic device handling."""
        # Shard batch across devices
        sharded_batch = jax.tree_map(
            lambda x: x.reshape((jax.device_count(), -1) + x.shape[1:]),
            batch
        )
        
        # Run training step
        outputs, new_state = self._pmapped_train_step(state, sharded_batch, rng)
        
        # Combine results from devices
        outputs = jax.device_get(outputs)
        
        return outputs, new_state
