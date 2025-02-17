import os
import json
import logging
from pathlib import Path
from typing import Dict, Union, Any, Optional, Tuple, List
from dataclasses import dataclass
import gc

import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import PreTrainedModel, Trainer, TrainingArguments

from .model import Transformer, ModelArgs
from .neural_memory import NeuralMemory
from .tree_of_thoughts import TreeOfThoughts
from .cache_augmentation import CacheAugmentation
from .reward_function import RewardConfig, RewardNetwork, SLAP, RewardTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class TrainingState:
    """Tracks training state and metrics."""
    steps: int = 0
    loss_history: List[float] = None
    memory_usage: List[Dict[str, float]] = None
    component_losses: Dict[str, List[float]] = None
    best_loss: float = float('inf')
    last_save_step: int = 0
    
    def __post_init__(self):
        self.loss_history = []
        self.memory_usage = []
        self.component_losses = {
            'memory': [],
            'tree': [],
            'cache': [],
            'reward': []
        }
    
    def update(self, step_stats: Dict[str, Any]):
        """Update training state with new statistics."""
        self.steps += 1
        self.loss_history.append(step_stats['total_loss'])
        self.memory_usage.append(step_stats['memory_usage'])
        
        for component in self.component_losses:
            if f'{component}_loss' in step_stats:
                self.component_losses[component].append(
                    step_stats[f'{component}_loss']
                )
                
        if step_stats['total_loss'] < self.best_loss:
            self.best_loss = step_stats['total_loss']
            logger.info(f"New best loss: {self.best_loss:.4f}")

def clear_gpu_memory():
    """Clear GPU memory cache and collect garbage."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

class VishwamAIPretrainer(Trainer):
    """
    Enhanced trainer incorporating memory, tree search, cache augmentation,
    and reward-based learning.
    """
    
    def __init__(
        self,
        model_config: Dict[str, Any],
        memory_module: Optional[NeuralMemory] = None,
        tree_module: Optional[TreeOfThoughts] = None,
        cache_module: Optional[CacheAugmentation] = None,
        reward_config: Optional[RewardConfig] = None,
        checkpoint_dir: Optional[str] = None,
        **kwargs
    ):
        """
        Initialize trainer with model and components.
        
        Args:
            model_config: Model configuration dictionary
            memory_module: Optional neural memory module
            tree_module: Optional tree of thoughts module
            cache_module: Optional cache augmentation module
            reward_config: Optional reward configuration
            checkpoint_dir: Directory for saving checkpoints
            **kwargs: Additional arguments passed to base Trainer
        """
        super().__init__(**kwargs)
        
        self.model_config = model_config
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        
        # Initialize training state
        self.state = TrainingState()
        
        try:
            self.initialize_components()
            
            # Initialize modules
            self.memory_module = memory_module
            self.tree_module = tree_module
            self.cache_module = cache_module
            
            # Initialize reward components if config provided
            self._init_reward_components(reward_config)
            
            self.scaler = GradScaler()
            self.gradient_accumulation_steps = kwargs.get(
                'gradient_accumulation_steps',
                1
            )
            
            # Enable FSDP if in distributed training
            if dist.is_initialized():
                self._setup_distributed()
                
        except Exception as e:
            logger.error(f"Failed to initialize trainer: {str(e)}")
            raise
            
        logger.info("Trainer initialized successfully")

    def _init_reward_components(self, reward_config: Optional[RewardConfig]):
        """Initialize reward-based training components."""
        if reward_config is not None:
            try:
                reward_net = RewardNetwork(reward_config)
                self.slap_module = SLAP(reward_net)
                self.reward_trainer = RewardTrainer(self.slap_module)
            except Exception as e:
                logger.error(f"Failed to initialize reward components: {str(e)}")
                self.slap_module = None
                self.reward_trainer = None
        else:
            self.slap_module = None
            self.reward_trainer = None

    def initialize_components(self):
        """Initialize model and components with proper configuration."""
        logger.info("Initializing model and components...")
        clear_gpu_memory()

        try:
            # Initialize main model with provided configuration
            model_args = ModelArgs(**self.model_config)
            self.model = Transformer(model_args)
            logger.info("Model initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize model: {str(e)}")
            raise

    def _setup_distributed(self):
        """Configure model for distributed training."""
        try:
            self.model = FSDP(
                self.model,
                auto_wrap_policy=transformer_auto_wrap_policy,
                mixed_precision=True
            )
            logger.info("Distributed training setup complete")
            
        except Exception as e:
            logger.error(f"Failed to setup distributed training: {str(e)}")
            raise

    def compute_loss(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor],
        return_outputs: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """
        Custom loss computation incorporating all components.
        
        Args:
            model: The model to train
            inputs: Input tensors
            return_outputs: Whether to return model outputs
            
        Returns:
            Loss tensor or tuple of (loss, outputs)
        """
        try:
            # Track component losses
            losses = {}
            
            # Get base model outputs
            outputs = model(**inputs)
            base_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
            losses['base'] = base_loss.item()
            
            total_loss = base_loss
            hidden_states = (
                outputs.hidden_states[-1]
                if outputs.hidden_states is not None
                else None
            )

            if hidden_states is not None:
                # Apply memory enhancement
                if self.memory_module is not None:
                    memory_enhanced = self.memory_module(hidden_states)
                    memory_loss = self.compute_auxiliary_loss(
                        memory_enhanced,
                        hidden_states,
                        'memory'
                    )
                    total_loss = total_loss + memory_loss
                    losses['memory'] = memory_loss.item()

                # Apply tree of thoughts
                if self.tree_module is not None:
                    tree_enhanced = self.tree_module(hidden_states)
                    tree_loss = self.compute_auxiliary_loss(
                        tree_enhanced,
                        hidden_states,
                        'tree'
                    )
                    total_loss = total_loss + tree_loss
                    losses['tree'] = tree_loss.item()

                # Apply cache augmentation
                if self.cache_module is not None:
                    cache_enhanced = self.cache_module(hidden_states)
                    cache_loss = self.compute_auxiliary_loss(
                        cache_enhanced,
                        hidden_states,
                        'cache'
                    )
                    total_loss = total_loss + cache_loss
                    losses['cache'] = cache_loss.item()
                    
                # Add reward computation
                if self.slap_module is not None:
                    rewards, action_loss = self.slap_module(hidden_states)
                    reward_loss = sum(
                        r.mean() for name, r in rewards.items()
                        if name != 'value'
                    )
                    if action_loss is not None:
                        reward_loss = reward_loss + 0.1 * action_loss
                    total_loss = total_loss + reward_loss
                    losses['reward'] = reward_loss.item()

            # Log component losses
            self._log_losses(losses)

            if return_outputs:
                return total_loss, outputs
            return total_loss
            
        except Exception as e:
            logger.error(f"Error computing loss: {str(e)}")
            raise

    def compute_auxiliary_loss(
        self,
        enhanced_states: torch.Tensor,
        original_states: torch.Tensor,
        component: str
    ) -> torch.Tensor:
        """
        Compute auxiliary loss between enhanced and original states.
        
        Args:
            enhanced_states: Enhanced hidden states
            original_states: Original hidden states
            component: Name of component for loss weighting
            
        Returns:
            Combined auxiliary loss
        """
        try:
            # Component-specific loss weights
            weights = {
                'memory': 0.1,
                'tree': 0.2,
                'cache': 0.15
            }
            weight = weights.get(component, 0.1)
            
            # Contrastive loss
            cos_sim = F.cosine_similarity(
                enhanced_states,
                original_states,
                dim=-1
            )
            contrastive_loss = -torch.log(
                torch.exp(cos_sim) / torch.exp(cos_sim).sum()
            )
            
            # L2 regularization
            reg_loss = 0.01 * (
                enhanced_states.pow(2).mean() +
                original_states.pow(2).mean()
            )
            
            return weight * (contrastive_loss.mean() + reg_loss)
            
        except Exception as e:
            logger.error(
                f"Error computing auxiliary loss for {component}: {str(e)}"
            )
            raise

    def training_step(
        self,
        model: PreTrainedModel,
        inputs: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """
        Custom training step with device management and gradient scaling.
        
        Args:
            model: The model to train
            inputs: Input tensors
            
        Returns:
            Loss tensor
        """
        try:
            model.train()
            inputs = self._prepare_inputs(inputs)

            # Move modules to correct device
            device = inputs[list(inputs.keys())[0]].device
            self._move_modules_to_device(device)

            # Compute loss with mixed precision
            with autocast():
                loss = self.compute_loss(model, inputs)
                
            # Scale loss and backward pass
            self.scaler.scale(loss).backward()
            
            # Update weights if gradient accumulation complete
            if self.state.steps % self.gradient_accumulation_steps == 0:
                self._update_weights(model)
                
            # Update training state
            loss_val = loss.detach().item()
            self._update_training_state({
                'total_loss': loss_val,
                'memory_usage': self._get_memory_usage()
            })
            
            # Save checkpoint if needed
            self._maybe_save_checkpoint()
            
            return loss.detach()
            
        except Exception as e:
            logger.error(f"Error in training step: {str(e)}")
            raise

    def _move_modules_to_device(self, device: torch.device):
        """Move all modules to specified device."""
        for module in [self.memory_module, self.tree_module, self.cache_module]:
            if module is not None:
                module.train()
                module.to(device)

    def _update_weights(self, model: PreTrainedModel):
        """Update model weights with gradient scaling."""
        # Unscale gradients and clip
        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        # Optimizer step with scaler
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad()

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage statistics."""
        if torch.cuda.is_available():
            return {
                'allocated': torch.cuda.memory_allocated() / 1024**2,  # MB
                'reserved': torch.cuda.memory_reserved() / 1024**2,  # MB
                'max_allocated': torch.cuda.max_memory_allocated() / 1024**2  # MB
            }
        return {}

    def _log_losses(self, losses: Dict[str, float]):
        """Log component-wise losses."""
        log_str = " | ".join(
            f"{name.capitalize()} Loss: {val:.4f}"
            for name, val in losses.items()
        )
        logger.info(f"Step {self.state.steps} | {log_str}")

    def _update_training_state(self, stats: Dict[str, Any]):
        """Update training state with new statistics."""
        self.state.update(stats)

    def _maybe_save_checkpoint(self):
        """Save checkpoint if conditions are met."""
        if self.checkpoint_dir is None:
            return
            
        save_steps = 1000  # Save every 1000 steps
        if (self.state.steps - self.state.last_save_step) >= save_steps:
            self.save_model(str(self.checkpoint_dir / f"step_{self.state.steps}"))
            self.state.last_save_step = self.state.steps

    def save_model(
        self,
        output_dir: Optional[str] = None,
        _internal_call: bool = False
    ):
        """
        Save model and all components.
        
        Args:
            output_dir: Directory to save to
            _internal_call: Whether this is an internal save call
        """
        try:
            # Create output directory
            if output_dir is not None:
                os.makedirs(output_dir, exist_ok=True)
                
                # Save main model
                super().save_model(output_dir, _internal_call)
                
                # Save additional components
                for name, module in [
                    ('memory', self.memory_module),
                    ('tree', self.tree_module),
                    ('cache', self.cache_module)
                ]:
                    if module is not None:
                        module_dir = os.path.join(output_dir, name)
                        os.makedirs(module_dir, exist_ok=True)
                        module.save_pretrained(module_dir)
                        
                # Save training state
                state_path = os.path.join(output_dir, 'training_state.json')
                with open(state_path, 'w') as f:
                    json.dump({
                        'steps': self.state.steps,
                        'best_loss': self.state.best_loss,
                        'last_save_step': self.state.last_save_step
                    }, f)
                    
                logger.info(f"Model and components saved to {output_dir}")
                
        except Exception as e:
            logger.error(f"Error saving model: {str(e)}")
            raise

    def load_model(self, input_dir: str):
        """
        Load model and all components.
        
        Args:
            input_dir: Directory to load from
        """
        try:
            # Load main model
            super().load_model(input_dir)
            
            # Load additional components
            for name, module in [
                ('memory', self.memory_module),
                ('tree', self.tree_module),
                ('cache', self.cache_module)
            ]:
                module_dir = os.path.join(input_dir, name)
                if module is not None and os.path.exists(module_dir):
                    module.load_pretrained(module_dir)
                    
            # Load training state
            state_path = os.path.join(input_dir, 'training_state.json')
            if os.path.exists(state_path):
                with open(state_path) as f:
                    state_dict = json.load(f)
                    self.state.steps = state_dict['steps']
                    self.state.best_loss = state_dict['best_loss']
                    self.state.last_save_step = state_dict['last_save_step']
                    
            logger.info(f"Model and components loaded from {input_dir}")
            
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
