"""Qwen-specific distillation implementation with gradient accumulation."""
import jax
import jax.numpy as jnp
from functools import partial
from typing import Dict, Any, Optional, Tuple
from flax.training import train_state
import optax

from .distillation import VishwamaiShaalaTrainer
from .layer_mapping import create_feature_mapping_config
from .loss_functions import compute_distillation_loss

class QwenDistillationTrainer(VishwamaiShaalaTrainer):
    """Specialized trainer for Qwen model distillation with gradient accumulation."""

    def __init__(self, teacher_model, student_model, cfg):
        super().__init__(teacher_model, student_model, cfg)
        self.feature_mapping = create_feature_mapping_config(
            student_config=self.cfg.distillation.student_model.config,
            teacher_config=self.cfg.distillation.teacher_model.config
        )
        
        if self.feature_mapping['hidden_mapping']['needs_projection']:
            self._initialize_projection_layers()
    
    def _initialize_projection_layers(self):
        """Initialize projection layers for dimension matching."""
        student_dim = self.feature_mapping['hidden_mapping']['student_dim']
        teacher_dim = self.feature_mapping['hidden_mapping']['teacher_dim']
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

    def train_step_with_grads(
        self, 
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
        rng: jnp.ndarray
    ) -> Tuple[train_state.TrainState, Dict[str, Any], Dict[str, jnp.ndarray]]:
        """Training step that returns gradients for accumulation."""
        
        def loss_fn(params):
            # Forward passes
            student_outputs = self.student_model.apply(
                {'params': params},
                batch['input_ids'],
                attention_mask=batch.get('attention_mask'),
                output_hidden_states=True,
                output_attentions=True,
                deterministic=False,
                rngs={'dropout': rng}
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
            
            # Process and align model outputs
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
        
        # Compute loss and gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (metrics, outputs)), grads = grad_fn(state.params)
        
        return state, metrics, grads

    def _process_model_outputs(
        self, 
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process and align model outputs."""
        # Extract and align features
        student_features = {}
        teacher_features = {}
        
        # Process student features
        for idx, hidden in enumerate(student_outputs['hidden_states']):
            layer_key = f'student_layer_{idx}'
            student_features[layer_key] = (
                self._project_hidden_states(hidden)
                if self.feature_mapping['hidden_mapping']['needs_projection']
                else hidden
            )
        
        # Process teacher features
        for idx, hidden in enumerate(teacher_outputs['hidden_states']):
            layer_key = f'teacher_layer_{idx}'
            teacher_features[layer_key] = hidden
        
        # Process attention patterns if available
        if 'attentions' in student_outputs and 'attentions' in teacher_outputs:
            for idx, (student_attn, teacher_attn) in enumerate(
                zip(student_outputs['attentions'], teacher_outputs['attentions'])
            ):
                student_features[f'student_attention_{idx}'] = student_attn
                teacher_features[f'teacher_attention_{idx}'] = teacher_attn
        
        return {
            'student_logits': student_outputs['logits'],
            'teacher_logits': teacher_outputs['logits'],
            **student_features,
            **teacher_features
        }

    def create_train_state(self, rng: jnp.ndarray) -> train_state.TrainState:
        """Create initial training state with gradient accumulation support."""
        # Initialize model
        dummy_input = jnp.ones((1, 16), dtype=jnp.int32)
        init_variables = self.student_model.init(rng, dummy_input)
        
        # Create optimizer with gradient accumulation
        steps_per_update = (
            self.cfg.training.gradient_accumulation_steps *
            jax.device_count()
        )
        
        tx = optax.chain(
            optax.clip_by_global_norm(self.cfg.training.max_grad_norm),
            optax.adam(
                learning_rate=self.cfg.training.learning_rate,
                b1=0.9,
                b2=0.999
            ),
            optax.update_every(steps_per_update)  # Accumulate for specified steps
        )
        
        return train_state.TrainState.create(
            apply_fn=self.student_model.__call__,
            params=init_variables['params'],
            tx=tx,
        )

if __name__ == "__main__":
    # Test usage
    from omegaconf import OmegaConf
    
    cfg = OmegaConf.create({
        'training': {
            'learning_rate': 1e-4,
            'max_grad_norm': 1.0,
            'gradient_accumulation_steps': 16
        },
        'distillation_params': {
            'temperature': 2.0,
            'alpha_ce': 0.2,
            'alpha_kd': 0.8,
            'feature_loss_weight': 0.1,
            'attention_loss_weight': 0.1
        }
    })
    
    trainer = QwenDistillationTrainer(None, None, cfg)
    print("Trainer initialized successfully")
