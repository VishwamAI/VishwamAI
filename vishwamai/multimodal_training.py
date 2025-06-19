"""
Multimodal Development Pipeline - Gemma Inspired

This module provides a complete development pipeline for multimodal models
with curriculum learning, progressive training, and advanced optimization
techniques inspired by Google DeepMind's Gemma architecture.

Features:
- Progressive multimodal training
- Curriculum learning for different modalities
- Advanced optimization with adaptive learning rates
- Multi-stage training with vision-text alignment
- Memory-efficient training with gradient checkpointing
- Distributed training support
- Model scaling and adaptation techniques
"""

import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state, checkpoints
from flax.core import freeze, unfreeze
import optax
from typing import Dict, Any, Optional, Tuple, List, Callable
import chex
import math
import functools
from dataclasses import dataclass, field
from pathlib import Path
import logging

from .advanced_multimodal import (
    GemmaInspiredMultimodalTransformer, 
    MultimodalConfig,
    GEMMA_4B_MULTIMODAL_CONFIG,
    GEMMA_12B_MULTIMODAL_CONFIG
)
from .gemma_attention import FlashAttention2, CrossModalAttention


@dataclass
class TrainingConfig:
    """Configuration for multimodal training pipeline."""
    
    # Model configuration
    model_config: MultimodalConfig = field(default_factory=lambda: GEMMA_4B_MULTIMODAL_CONFIG)
    
    # Training hyperparameters
    learning_rate: float = 1e-4
    warmup_steps: int = 4000
    max_steps: int = 100000
    batch_size: int = 32
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_stages: List[Dict[str, Any]] = field(default_factory=lambda: [
        {
            "name": "text_only",
            "steps": 10000,
            "modalities": ["text"],
            "learning_rate_multiplier": 1.0
        },
        {
            "name": "vision_alignment", 
            "steps": 20000,
            "modalities": ["text", "vision"],
            "learning_rate_multiplier": 0.5,
            "vision_loss_weight": 2.0
        },
        {
            "name": "multimodal_finetuning",
            "steps": 70000,
            "modalities": ["text", "vision"],
            "learning_rate_multiplier": 0.1,
            "cross_attention_weight": 1.5
        }
    ])
    
    # Data configuration
    max_seq_length: int = 2048
    image_size: int = 800
    vision_augmentation: bool = True
    text_augmentation: bool = False
    
    # Optimization
    optimizer: str = "adamw"  # "adamw", "adafactor", "lion"
    beta1: float = 0.9
    beta2: float = 0.95
    eps: float = 1e-8
    use_gradient_checkpointing: bool = True
    use_mixed_precision: bool = True
    
    # Regularization
    dropout: float = 0.1
    attention_dropout: float = 0.1
    label_smoothing: float = 0.1
    
    # Checkpointing and logging
    checkpoint_dir: str = "./checkpoints"
    save_every_steps: int = 1000
    eval_every_steps: int = 500
    log_every_steps: int = 100
    
    # Hardware optimization
    use_sharding: bool = True
    mesh_shape: Tuple[int, ...] = (1, 1)  # (data_parallel, model_parallel)
    
    # Experimental features
    use_flash_attention: bool = True
    use_soft_capping: bool = True
    use_rope_scaling: bool = True


class CurriculumStage:
    """Represents a single stage in curriculum learning."""
    
    def __init__(
        self,
        name: str,
        steps: int,
        modalities: List[str],
        learning_rate_multiplier: float = 1.0,
        **kwargs
    ):
        self.name = name
        self.steps = steps
        self.modalities = modalities
        self.learning_rate_multiplier = learning_rate_multiplier
        self.config = kwargs
    
    def should_use_modality(self, modality: str) -> bool:
        """Check if this stage uses the given modality."""
        return modality in self.modalities
    
    def get_loss_weight(self, loss_type: str) -> float:
        """Get weight for specific loss type in this stage."""
        return self.config.get(f"{loss_type}_weight", 1.0)


class AdaptiveLearningRateSchedule:
    """Adaptive learning rate schedule with warmup and decay."""
    
    def __init__(
        self,
        base_lr: float,
        warmup_steps: int,
        max_steps: int,
        min_lr_ratio: float = 0.1,
        decay_type: str = "cosine"
    ):
        self.base_lr = base_lr
        self.warmup_steps = warmup_steps
        self.max_steps = max_steps
        self.min_lr_ratio = min_lr_ratio
        self.decay_type = decay_type
    
    def __call__(self, step: int) -> float:
        """Compute learning rate for given step."""
        
        if step < self.warmup_steps:
            # Linear warmup
            return self.base_lr * (step / self.warmup_steps)
        
        # Decay phase
        progress = (step - self.warmup_steps) / (self.max_steps - self.warmup_steps)
        progress = jnp.clip(progress, 0.0, 1.0)
        
        if self.decay_type == "cosine":
            decay_factor = 0.5 * (1 + jnp.cos(jnp.pi * progress))
        elif self.decay_type == "linear":
            decay_factor = 1.0 - progress
        elif self.decay_type == "polynomial":
            decay_factor = (1.0 - progress) ** 2
        else:
            raise ValueError(f"Unknown decay type: {self.decay_type}")
        
        min_lr = self.base_lr * self.min_lr_ratio
        return min_lr + (self.base_lr - min_lr) * decay_factor


class MultimodalLoss:
    """Advanced loss computation for multimodal training."""
    
    def __init__(
        self,
        vocab_size: int,
        label_smoothing: float = 0.1,
        ignore_index: int = -100
    ):
        self.vocab_size = vocab_size
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
    
    def cross_entropy_loss(
        self,
        logits: chex.Array,
        labels: chex.Array,
        weights: Optional[chex.Array] = None
    ) -> chex.Array:
        """Compute cross-entropy loss with label smoothing."""
        
        # Apply label smoothing
        if self.label_smoothing > 0:
            smooth_labels = optax.smooth_labels(
                labels, alpha=self.label_smoothing
            )
        else:
            smooth_labels = labels
        
        # Compute cross-entropy
        loss = optax.softmax_cross_entropy(logits, smooth_labels)
        
        # Apply weights if provided
        if weights is not None:
            loss = loss * weights
        
        # Mask out ignored tokens
        if self.ignore_index is not None:
            mask = labels != self.ignore_index
            loss = loss * mask
            return jnp.sum(loss) / jnp.sum(mask)
        
        return jnp.mean(loss)
    
    def contrastive_loss(
        self,
        text_embeddings: chex.Array,
        vision_embeddings: chex.Array,
        temperature: float = 0.07
    ) -> chex.Array:
        """Compute contrastive loss for vision-text alignment."""
        
        # Normalize embeddings
        text_embeddings = text_embeddings / jnp.linalg.norm(
            text_embeddings, axis=-1, keepdims=True
        )
        vision_embeddings = vision_embeddings / jnp.linalg.norm(
            vision_embeddings, axis=-1, keepdims=True
        )
        
        # Compute similarity matrix
        similarity = jnp.dot(text_embeddings, vision_embeddings.T) / temperature
        
        # Create labels (diagonal is positive pairs)
        batch_size = text_embeddings.shape[0]
        labels = jnp.arange(batch_size)
        
        # Compute symmetric cross-entropy loss
        text_loss = optax.softmax_cross_entropy_with_integer_labels(
            similarity, labels
        )
        vision_loss = optax.softmax_cross_entropy_with_integer_labels(
            similarity.T, labels
        )
        
        return (jnp.mean(text_loss) + jnp.mean(vision_loss)) / 2
    
    def compute_total_loss(
        self,
        outputs: Dict[str, chex.Array],
        targets: Dict[str, chex.Array],
        stage: CurriculumStage,
        step: int
    ) -> Tuple[chex.Array, Dict[str, chex.Array]]:
        """Compute total loss for current training stage."""
        
        losses = {}
        total_loss = 0.0
        
        # Language modeling loss
        if "logits" in outputs and "labels" in targets:
            lm_loss = self.cross_entropy_loss(
                outputs["logits"],
                targets["labels"],
                targets.get("loss_mask")
            )
            losses["language_modeling"] = lm_loss
            total_loss += lm_loss * stage.get_loss_weight("language_modeling")
        
        # Vision-text contrastive loss
        if (stage.should_use_modality("vision") and 
            "text_embeddings" in outputs and 
            "vision_embeddings" in outputs):
            
            contrastive_loss = self.contrastive_loss(
                outputs["text_embeddings"],
                outputs["vision_embeddings"]
            )
            losses["contrastive"] = contrastive_loss
            total_loss += contrastive_loss * stage.get_loss_weight("vision")
        
        # Cross-attention alignment loss
        if ("cross_attention_weights" in outputs and 
            stage.get_loss_weight("cross_attention") > 0):
            
            # Encourage attention to focus on relevant regions
            attn_weights = outputs["cross_attention_weights"]
            # Compute attention entropy to encourage focused attention
            attn_entropy = -jnp.sum(
                attn_weights * jnp.log(attn_weights + 1e-8), axis=-1
            )
            attn_loss = jnp.mean(attn_entropy)
            
            losses["cross_attention"] = attn_loss
            total_loss += attn_loss * stage.get_loss_weight("cross_attention")
        
        losses["total"] = total_loss
        return total_loss, losses


class MultimodalTrainer:
    """Main trainer class for multimodal models."""
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.model_config = config.model_config
        
        # Initialize curriculum stages
        self.curriculum_stages = [
            CurriculumStage(**stage_config)
            for stage_config in config.curriculum_stages
        ]
        
        # Initialize learning rate schedule
        self.lr_schedule = AdaptiveLearningRateSchedule(
            base_lr=config.learning_rate,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps
        )
        
        # Initialize loss function
        self.loss_fn = MultimodalLoss(
            vocab_size=self.model_config.vocab_size,
            label_smoothing=config.label_smoothing
        )
        
        # Initialize model
        self.model = GemmaInspiredMultimodalTransformer(
            config=self.model_config
        )
        
        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
    
    def create_optimizer(self) -> optax.GradientTransformation:
        """Create optimizer with learning rate schedule."""
        
        # Learning rate schedule
        lr_schedule = optax.schedule.inject_hyperparams(self.lr_schedule)
        
        # Choose optimizer
        if self.config.optimizer == "adamw":
            optimizer = optax.adamw(
                learning_rate=lr_schedule,
                b1=self.config.beta1,
                b2=self.config.beta2,
                eps=self.config.eps,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer == "adafactor":
            optimizer = optax.adafactor(
                learning_rate=lr_schedule,
                beta2_decay=-0.8,
                factored=True,
                min_dim_size_to_factor=128
            )
        elif self.config.optimizer == "lion":
            optimizer = optax.lion(
                learning_rate=lr_schedule,
                b1=self.config.beta1,
                b2=self.config.beta2,
                weight_decay=self.config.weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer: {self.config.optimizer}")
        
        # Add gradient clipping
        if self.config.max_grad_norm > 0:
            optimizer = optax.chain(
                optax.clip_by_global_norm(self.config.max_grad_norm),
                optimizer
            )
        
        # Add gradient accumulation if needed
        if self.config.gradient_accumulation_steps > 1:
            optimizer = optax.MultiSteps(
                optimizer, 
                every_k_schedule=self.config.gradient_accumulation_steps
            )
        
        return optimizer
    
    def get_current_stage(self, step: int) -> CurriculumStage:
        """Get current curriculum stage for given step."""
        
        if not self.config.use_curriculum:
            return self.curriculum_stages[-1]  # Use final stage
        
        cumulative_steps = 0
        for stage in self.curriculum_stages:
            cumulative_steps += stage.steps
            if step < cumulative_steps:
                return stage
        
        return self.curriculum_stages[-1]  # Use final stage if exceeded
    
    def compute_loss(
        self,
        params: chex.ArrayTree,
        batch: Dict[str, chex.Array],
        step: int,
        training: bool = True
    ) -> Tuple[chex.Array, Tuple[Dict[str, chex.Array], chex.ArrayTree]]:
        """Compute loss for a batch."""
        
        # Get current curriculum stage
        stage = self.get_current_stage(step)
        
        # Prepare inputs based on current stage
        model_inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch.get("attention_mask"),
            "training": training
        }
        
        # Add vision inputs if current stage uses vision
        if stage.should_use_modality("vision") and "images" in batch:
            model_inputs["images"] = batch["images"]
        
        # Forward pass
        outputs = self.model.apply(
            {"params": params},
            **model_inputs,
            rngs={"dropout": jax.random.PRNGKey(step)} if training else None
        )
        
        # Prepare targets
        targets = {
            "labels": batch.get("labels", batch["input_ids"]),
            "loss_mask": batch.get("loss_mask")
        }
        
        # Compute loss
        total_loss, losses = self.loss_fn.compute_total_loss(
            outputs={"logits": outputs},
            targets=targets,
            stage=stage,
            step=step
        )
        
        return total_loss, (losses, outputs)
    
    def train_step(
        self,
        state: train_state.TrainState,
        batch: Dict[str, chex.Array],
        step: int
    ) -> Tuple[train_state.TrainState, Dict[str, chex.Array]]:
        """Single training step."""
        
        def loss_fn(params):
            return self.compute_loss(params, batch, step, training=True)
        
        # Compute gradients
        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (losses, outputs)), grads = grad_fn(state.params)
        
        # Apply gradients
        state = state.apply_gradients(grads=grads)
        
        # Prepare metrics
        metrics = {
            "step": step,
            "learning_rate": self.lr_schedule(step),
            "total_loss": loss,
            **losses
        }
        
        # Add gradient norms
        grad_norm = optax.global_norm(grads)
        metrics["grad_norm"] = grad_norm
        
        return state, metrics
    
    def eval_step(
        self,
        params: chex.ArrayTree,
        batch: Dict[str, chex.Array],
        step: int
    ) -> Dict[str, chex.Array]:
        """Single evaluation step."""
        
        loss, (losses, outputs) = self.compute_loss(
            params, batch, step, training=False
        )
        
        # Compute additional metrics
        metrics = {
            "eval_loss": loss,
            **{f"eval_{k}": v for k, v in losses.items()}
        }
        
        # Add perplexity
        if "language_modeling" in losses:
            perplexity = jnp.exp(losses["language_modeling"])
            metrics["eval_perplexity"] = perplexity
        
        return metrics
    
    def initialize_training(
        self,
        dummy_batch: Dict[str, chex.Array]
    ) -> train_state.TrainState:
        """Initialize training state."""
        
        # Initialize model parameters
        rng = jax.random.PRNGKey(42)
        variables = self.model.init(
            rng,
            input_ids=dummy_batch["input_ids"],
            images=dummy_batch.get("images"),
            training=False
        )
        
        # Create optimizer
        optimizer = self.create_optimizer()
        
        # Create training state
        state = train_state.TrainState.create(
            apply_fn=self.model.apply,
            params=variables["params"],
            tx=optimizer
        )
        
        return state
    
    def train(
        self,
        train_loader: Any,
        eval_loader: Optional[Any] = None,
        resume_from_checkpoint: Optional[str] = None
    ) -> train_state.TrainState:
        """Main training loop."""
        
        # Get dummy batch for initialization
        dummy_batch = next(iter(train_loader))
        
        # Initialize training state
        if resume_from_checkpoint:
            state = checkpoints.restore_checkpoint(
                resume_from_checkpoint, target=None
            )
        else:
            state = self.initialize_training(dummy_batch)
        
        self.logger.info(f"Starting training from step {state.step}")
        self.logger.info(f"Model parameters: {sum(x.size for x in jax.tree_util.tree_leaves(state.params)):,}")
        
        # Training loop
        step = int(state.step)
        
        while step < self.config.max_steps:
            # Get current curriculum stage
            current_stage = self.get_current_stage(step)
            
            for batch in train_loader:
                if step >= self.config.max_steps:
                    break
                
                # Training step
                state, train_metrics = self.train_step(state, batch, step)
                
                # Logging
                if step % self.config.log_every_steps == 0:
                    self.logger.info(
                        f"Step {step} | Stage: {current_stage.name} | "
                        f"Loss: {train_metrics['total_loss']:.4f} | "
                        f"LR: {train_metrics['learning_rate']:.2e}"
                    )
                
                # Evaluation
                if (eval_loader is not None and 
                    step % self.config.eval_every_steps == 0):
                    
                    eval_metrics = {}
                    eval_steps = 0
                    
                    for eval_batch in eval_loader:
                        batch_metrics = self.eval_step(
                            state.params, eval_batch, step
                        )
                        
                        # Accumulate metrics
                        for key, value in batch_metrics.items():
                            if key not in eval_metrics:
                                eval_metrics[key] = 0.0
                            eval_metrics[key] += value
                        
                        eval_steps += 1
                        
                        # Limit evaluation steps
                        if eval_steps >= 100:
                            break
                    
                    # Average metrics
                    eval_metrics = {
                        k: v / eval_steps for k, v in eval_metrics.items()
                    }
                    
                    self.logger.info(
                        f"Eval Step {step} | "
                        f"Eval Loss: {eval_metrics['eval_loss']:.4f}"
                    )
                
                # Checkpointing
                if step % self.config.save_every_steps == 0:
                    checkpoint_path = Path(self.config.checkpoint_dir) / f"checkpoint_{step}"
                    checkpoints.save_checkpoint(
                        checkpoint_path,
                        state,
                        step=step,
                        keep=3
                    )
                    self.logger.info(f"Saved checkpoint at step {step}")
                
                step += 1
        
        self.logger.info("Training completed!")
        return state


def create_data_loader(
    dataset: Any,
    tokenizer: Any,
    config: TrainingConfig,
    training: bool = True
) -> Any:
    """Create data loader for multimodal training."""
    
    def collate_fn(batch):
        """Collate function for multimodal batches."""
        
        # Extract text and images
        texts = [item["text"] for item in batch]
        images = [item.get("image") for item in batch if item.get("image") is not None]
        
        # Tokenize text
        tokenized = tokenizer(
            texts,
            max_length=config.max_seq_length,
            padding=True,
            truncation=True,
            return_tensors="np"
        )
        
        batch_dict = {
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
            "labels": tokenized["input_ids"]  # For language modeling
        }
        
        # Add images if available
        if images:
            # Process images (resize, normalize, etc.)
            processed_images = jnp.array(images)  # Placeholder
            batch_dict["images"] = processed_images
        
        return batch_dict
    
    # Create data loader (implementation depends on your data framework)
    # This is a placeholder - replace with actual data loading logic
    return dataset  # Placeholder


def main_training_pipeline(
    train_dataset: Any,
    eval_dataset: Optional[Any] = None,
    tokenizer: Any = None,
    config: Optional[TrainingConfig] = None
) -> train_state.TrainState:
    """Main training pipeline function."""
    
    if config is None:
        config = TrainingConfig()
    
    # Create trainer
    trainer = MultimodalTrainer(config)
    
    # Create data loaders
    train_loader = create_data_loader(train_dataset, tokenizer, config, training=True)
    eval_loader = None
    if eval_dataset is not None:
        eval_loader = create_data_loader(eval_dataset, tokenizer, config, training=False)
    
    # Start training
    final_state = trainer.train(train_loader, eval_loader)
    
    return final_state


# Configuration presets for different model sizes
SMALL_MODEL_TRAINING_CONFIG = TrainingConfig(
    model_config=MultimodalConfig(
        embed_dim=1024,
        num_heads=8,
        num_kv_heads=4,
        vision_embed_dim=512,
        vision_layers=12
    ),
    learning_rate=2e-4,
    batch_size=64,
    max_steps=50000
)

MEDIUM_MODEL_TRAINING_CONFIG = TrainingConfig(
    model_config=GEMMA_4B_MULTIMODAL_CONFIG,
    learning_rate=1e-4,
    batch_size=32,
    max_steps=100000
)

LARGE_MODEL_TRAINING_CONFIG = TrainingConfig(
    model_config=GEMMA_12B_MULTIMODAL_CONFIG,
    learning_rate=5e-5,
    batch_size=16,
    max_steps=200000,
    gradient_accumulation_steps=4
)
