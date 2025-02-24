"""Knowledge distillation module for VishwamAI."""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from typing import Dict, List, Optional, Tuple, Any
from omegaconf import DictConfig

from .model import VishwamAIModel, ModelConfig

class VishwamaiGuruKnowledge:
    """Implements knowledge transfer from guru models (like OpenAI ChatGPT) to shishya (student) models.
    
    With a touch of humor, this module follows the ancient Indian guru-shishya parampara (teacher-student tradition)
    where GPT is like the wise guru sharing its deep wisdom with our eager Vishwamai shishya.
    """
    
    def __init__(self, cfg: DictConfig):
        """Initialize distillation parameters."""
        self.temperature = cfg.distillation.teacher_model.temperature
        self.alpha = cfg.distillation.teacher_model.alpha
        self.feature_layers = cfg.distillation.feature_distillation.layers
        self.feature_weight = cfg.distillation.feature_distillation.loss_weight
        self.attention_weight = cfg.distillation.attention_distillation.loss_weight
        self.hidden_weight = cfg.distillation.hidden_distillation.loss_weight

    def kl_div(
        self,
        student_logits: jnp.ndarray,
        teacher_logits: jnp.ndarray,
        temperature: float
    ) -> jnp.ndarray:
        """The divine knowledge gap calculator.
        
        Like measuring the distance between a wise OpenAI guru's knowledge
        and our eager Vishwamai student's understanding. The temperature
        controls how 'strict' our guru is - just as DeepSeek balances 
        precision with patience!
        """
        student_probs = jax.nn.softmax(student_logits / temperature, axis=-1)
        teacher_probs = jax.nn.softmax(teacher_logits / temperature, axis=-1)
        loss = jnp.sum(
            teacher_probs * (jnp.log(teacher_probs) - jnp.log(student_probs)),
            axis=-1
        )
        return loss.mean() * (temperature ** 2)

    def mse_loss(
        self,
        student_features: jnp.ndarray,
        teacher_features: jnp.ndarray
    ) -> jnp.ndarray:
        """The path deviation meter.
        
        Measures how far our shishya's understanding deviates from the 
        guru's wisdom. Like comparing our student's attempts with 
        ChatGPT's enlightened responses - every small deviation is a 
        learning opportunity!
        """
        return jnp.mean(jnp.square(student_features - teacher_features))

    def cosine_loss(
        self,
        student_features: jnp.ndarray,
        teacher_features: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute cosine similarity loss between student and teacher features."""
        student_norm = jnp.linalg.norm(student_features, axis=-1, keepdims=True)
        teacher_norm = jnp.linalg.norm(teacher_features, axis=-1, keepdims=True)
        cos_sim = jnp.sum(
            (student_features / student_norm) * (teacher_features / teacher_norm),
            axis=-1
        )
        return -jnp.mean(cos_sim)

    def attention_loss(
        self,
        student_attention: jnp.ndarray,
        teacher_attention: jnp.ndarray
    ) -> jnp.ndarray:
        """The focus alignment meter.
        
        Just as DeepSeek knows where to focus its attention, we guide
        our student to look at the right aspects of knowledge. Like a
        guru gently turning their shishya's gaze towards the important
        concepts, we align our model's attention with the masters.
        """
        # Normalize attention maps
        student_attention = student_attention / jnp.sqrt(
            jnp.sum(student_attention ** 2, axis=(-2, -1), keepdims=True)
        )
        teacher_attention = teacher_attention / jnp.sqrt(
            jnp.sum(teacher_attention ** 2, axis=(-2, -1), keepdims=True)
        )
        return jnp.mean(jnp.square(student_attention - teacher_attention))

    def compute_losses(
        self,
        student_outputs: Dict[str, Any],
        teacher_outputs: Dict[str, Any],
        labels: jnp.ndarray,
        mask: Optional[jnp.ndarray] = None
    ) -> Dict[str, jnp.ndarray]:
        """The great assessment of learning.
        
        Like the ancient tradition of guru-shishya evaluation, we measure how
        well our student (Vishwamai) has absorbed the wisdom of the great gurus
        (OpenAI and DeepSeek). We check knowledge (logits), understanding 
        (features), and attention (focus) - all aspects that make a wise AI!
        """
        losses = {}
        
        # Task-specific loss (e.g., cross-entropy)
        task_loss = student_outputs['loss']
        losses['task_loss'] = task_loss
        
        # Knowledge distillation loss
        kd_loss = self.kl_div(
            student_outputs['logits'],
            teacher_outputs['logits'],
            self.temperature
        )
        losses['kd_loss'] = kd_loss
        
        # Feature distillation loss if enabled
        if 'layer_outputs' in student_outputs and 'layer_outputs' in teacher_outputs:
            feature_losses = []
            for layer_idx in self.feature_layers:
                student_features = student_outputs['layer_outputs'][layer_idx]
                teacher_features = teacher_outputs['layer_outputs'][layer_idx]
                feature_losses.append(self.mse_loss(student_features, teacher_features))
            if feature_losses:
                losses['feature_loss'] = jnp.mean(jnp.stack(feature_losses))
        
        # Attention distillation loss if enabled
        if 'attention_maps' in student_outputs and 'attention_maps' in teacher_outputs:
            attention_losses = []
            for s_attn, t_attn in zip(
                student_outputs['attention_maps'],
                teacher_outputs['attention_maps']
            ):
                attention_losses.append(self.attention_loss(s_attn, t_attn))
            if attention_losses:
                losses['attention_loss'] = jnp.mean(jnp.stack(attention_losses))
        
        # Hidden state distillation loss if enabled
        if 'hidden_states' in student_outputs and 'hidden_states' in teacher_outputs:
            hidden_loss = self.mse_loss(
                student_outputs['hidden_states'],
                teacher_outputs['hidden_states']
            )
            losses['hidden_loss'] = hidden_loss
        
        # Combine all losses with their respective weights
        total_loss = (
            (1 - self.alpha) * task_loss +
            self.alpha * kd_loss +
            self.feature_weight * losses.get('feature_loss', 0.0) +
            self.attention_weight * losses.get('attention_loss', 0.0) +
            self.hidden_weight * losses.get('hidden_loss', 0.0)
        )
        losses['total_loss'] = total_loss
        
        return losses

class VishwamaiShaalaTrainer:
    """Manages the gurukul (ancient learning center) training process.
    
    Just as a guru teaches their shishya in the peaceful ambiance of a gurukul,
    this trainer helps transfer knowledge from wise models like DeepSeek and OpenAI
    to our humble Vishwamai student. As they say in Sanskrit - "Acharya Devo Bhava"
    (Teacher is Divine), we treat our teacher models with utmost respect while learning from them.
    """
    
    def __init__(
        self,
        teacher_model: VishwamAIModel,
        student_model: VishwamAIModel,
        cfg: DictConfig
    ):
        """Initialize teacher and student models."""
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.loss_fn = VishwamaiGuruKnowledge(cfg)
        self.cfg = cfg
        
        # Setup pruning if enabled
        self.pruning_enabled = cfg.distillation.pruning.enabled
        if self.pruning_enabled:
            self.pruning_schedule = self._create_pruning_schedule(
                cfg.distillation.pruning.target_sparsity,
                cfg.distillation.pruning.begin_step,
                cfg.distillation.pruning.end_step,
                cfg.distillation.pruning.pruning_schedule
            )
    
    def _create_pruning_schedule(
        self,
        target_sparsity: float,
        begin_step: int,
        end_step: int,
        schedule_type: str
    ) -> callable:
        """Create pruning schedule function."""
        if schedule_type == 'linear':
            def schedule(step):
                if step < begin_step:
                    return 0.0
                if step > end_step:
                    return target_sparsity
                return target_sparsity * (step - begin_step) / (end_step - begin_step)
        elif schedule_type == 'cubic':
            def schedule(step):
                if step < begin_step:
                    return 0.0
                if step > end_step:
                    return target_sparsity
                prog = (step - begin_step) / (end_step - begin_step)
                return target_sparsity * (prog ** 3)
        else:
            raise ValueError(f"Unknown pruning schedule: {schedule_type}")
        
        return schedule

    def train_step(
        self,
        state: train_state.TrainState,
        batch: Dict[str, jnp.ndarray],
        step: int,
        rng: jax.random.PRNGKey
    ) -> Tuple[train_state.TrainState, Dict[str, float], jax.random.PRNGKey]:
        """Execute single gurukul training step.
        
        Like a student practicing yoga asanas under the guru's guidance,
        our Vishwamai model refines its knowledge by learning from the great
        OpenAI and DeepSeek gurus. Each step is a small path to enlightenment!
        """
        
        def loss_fn(params):
            # Get student outputs
            student_outputs = self.student_model.apply(
                {'params': params},
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                train=True,
                capture_intermediates=True,
                rngs={'dropout': rng}
            )
            
            # Get teacher outputs (no gradient tracking needed)
            teacher_outputs = self.teacher_model.apply(
                {'params': state.params},
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                train=False,
                capture_intermediates=True
            )
            
            # Compute all distillation losses
            losses = self.loss_fn.compute_losses(
                student_outputs,
                teacher_outputs,
                batch['labels'],
                batch.get('attention_mask')
            )
            
            return losses['total_loss'], losses
        
        # Compute gradients and update model
        if state.dynamic_scale:
            # Handle mixed precision training
            dynamic_scale, is_finite, aux = state.dynamic_scale.value_and_grad(
                loss_fn, has_aux=True)(state.params)
            grad, losses = aux
            dynamic_scale = dynamic_scale.update(is_finite)
            state = state.replace(dynamic_scale=dynamic_scale)
            grad = jax.tree_map(
                lambda x: jnp.where(is_finite, x, jnp.zeros_like(x)),
                grad
            )
        else:
            (loss, losses), grad = jax.value_and_grad(
                loss_fn, has_aux=True)(state.params)
        
        # Apply pruning if enabled
        if self.pruning_enabled:
            sparsity = self.pruning_schedule(step)
            grad = self._apply_pruning_mask(grad, sparsity)
        
        # Update state
        state = state.apply_gradients(grads=grad)
        
        # Generate new RNG
        rng, new_rng = jax.random.split(rng)
        
        return state, losses, new_rng

    def _apply_pruning_mask(
        self,
        params: Dict[str, jnp.ndarray],
        sparsity: float
    ) -> Dict[str, jnp.ndarray]:
        """Apply the wisdom filter.
        
        Just as ancient texts speak of "Neti Neti" (not this, not this) to eliminate the non-essential,
        we use pruning to keep only the most enlightened neural pathways. Like OpenAI's GPT models
        that learn to be concise and DeepSeek's focused knowledge, we too seek the path of simplicity.
        """
        def prune_param(param):
            if len(param.shape) < 2:  # Skip 1D params like biases
                return param
            
            # Calculate threshold for magnitude pruning
            abs_param = jnp.abs(param)
            threshold = jnp.quantile(abs_param, sparsity)
            
            # Create binary mask
            mask = abs_param > threshold
            
            # Apply mask
            return param * mask
        
        return jax.tree_map(prune_param, params)

    def quantize_model(
        self,
        state: train_state.TrainState,
        val_loader: Any,
        num_calibration_steps: int
    ) -> train_state.TrainState:
        """The path of minimalism.
        
        Like the ancient yogis who learned to live with minimal possessions,
        we teach our Vishwamai model to attain the same wisdom as OpenAI and DeepSeek
        but with fewer bits. Through meditation (calibration) and self-discipline 
        (quantization), we achieve computational enlightenment with int8 precision!
        """
        if not self.cfg.distillation.quantization.enabled:
            return state
        
        # Collect activation statistics for calibration
        activation_stats = {}
        for _ in range(num_calibration_steps):
            batch = next(val_loader)
            outputs = self.student_model.apply(
                {'params': state.params},
                input_ids=batch['input_ids'],
                attention_mask=batch['attention_mask'],
                train=False,
                capture_intermediates=True
            )
            
            # Update activation statistics
            for layer_name, activations in outputs['activations'].items():
                if layer_name not in activation_stats:
                    activation_stats[layer_name] = []
                activation_stats[layer_name].append(activations)
        
        # Calculate quantization parameters based on collected statistics
        # This is a simplified example - in practice, you'd want to use a proper
        # quantization library like TensorRT or ONNX Runtime
        def quantize_params(param, name):
            if len(param.shape) < 2:  # Skip 1D params
                return param
                
            if self.cfg.distillation.quantization.precision == "int8":
                # Simple min-max quantization to int8
                param_min = jnp.min(param)
                param_max = jnp.max(param)
                scale = (param_max - param_min) / 255
                zero_point = -jnp.round(param_min / scale)
                
                # Quantize to int8
                quantized = jnp.round(param / scale + zero_point)
                quantized = jnp.clip(quantized, 0, 255)
                
                # Dequantize
                dequantized = (quantized - zero_point) * scale
                return dequantized
            else:
                raise ValueError(
                    f"Unsupported quantization precision: "
                    f"{self.cfg.distillation.quantization.precision}"
                )
        
        # Apply quantization to model parameters
        quantized_params = jax.tree_map(quantize_params, state.params)
        return state.replace(params=quantized_params)
