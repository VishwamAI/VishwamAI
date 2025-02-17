import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import PreTrainedModel, Trainer, TrainingArguments
from typing import Dict, Union, Any, Optional, Tuple
import gc

from .model import Transformer, ModelArgs
from .neural_memory import NeuralMemory
from .tree_of_thoughts import TreeOfThoughts
from .cache_augmentation import CacheAugmentation
from .reward_function import RewardConfig, RewardNetwork, SLAP, RewardTrainer

def clear_gpu_memory():
    """Clear GPU memory cache"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

class VishwamAIPretrainer(Trainer):
    def __init__(self, 
                 model_config: Dict[str, Any],
                 memory_module: Optional[NeuralMemory] = None,
                 tree_module: Optional[TreeOfThoughts] = None,
                 cache_module: Optional[CacheAugmentation] = None,
                 reward_config: Optional[RewardConfig] = None,
                 **kwargs):
        """Initialize trainer with model and components."""
        super().__init__(**kwargs)
        
        self.model_config = model_config
        self.initialize_components()
        
        # Initialize modules
        self.memory_module = memory_module
        self.tree_module = tree_module 
        self.cache_module = cache_module
        
        # Initialize reward components if config provided
        if reward_config is not None:
            reward_net = RewardNetwork(reward_config)
            self.slap_module = SLAP(reward_net)
            self.reward_trainer = RewardTrainer(self.slap_module)
        else:
            self.slap_module = None
            self.reward_trainer = None
        
        self.scaler = GradScaler()
        self.gradient_accumulation_steps = kwargs.get('gradient_accumulation_steps', 1)
        self.steps = 0  # Initialize step counter
        
        # Enable FSDP if in distributed training
        if dist.is_initialized():
            self.model = FSDP(
                self.model,
                auto_wrap_policy=transformer_auto_wrap_policy,
                mixed_precision=True
            )

    def initialize_components(self):
        """Initialize model and components with proper configuration."""
        print("Initializing model and components...")
        clear_gpu_memory()

        # Initialize main model with provided configuration
        model_args = ModelArgs(**self.model_config)
        self.model = Transformer(model_args)

    def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor], return_outputs: bool = False) -> Union[torch.Tensor, Tuple[torch.Tensor, Any]]:
        """Custom loss computation incorporating memory, tree search and cache."""
        
        # Get base model outputs
        outputs = model(**inputs)
        base_loss = outputs.loss if hasattr(outputs, 'loss') else outputs[0]
        
        total_loss = base_loss
        hidden_states = outputs.hidden_states[-1] if outputs.hidden_states is not None else None

        if hidden_states is not None:
            # Apply memory enhancement if available
            if self.memory_module is not None:
                memory_enhanced = self.memory_module(hidden_states)
                total_loss = total_loss + self.compute_auxiliary_loss(memory_enhanced, hidden_states)

            # Apply tree of thoughts if available
            if self.tree_module is not None:
                tree_enhanced = self.tree_module(hidden_states)
                total_loss = total_loss + self.compute_auxiliary_loss(tree_enhanced, hidden_states)

            # Apply cache augmentation if available
            if self.cache_module is not None:
                cache_enhanced = self.cache_module(hidden_states)
                total_loss = total_loss + self.compute_auxiliary_loss(cache_enhanced, hidden_states)
                
            # Add reward computation
            rewards, action_loss = self.slap_module(hidden_states)
            for reward_name, reward_value in rewards.items():
                if reward_name != 'value':
                    total_loss = total_loss + reward_value.mean()
            if action_loss is not None:
                total_loss = total_loss + 0.1 * action_loss

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def compute_auxiliary_loss(self, enhanced_states: torch.Tensor, original_states: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss between enhanced and original states."""
        # Add contrastive loss component
        cos_sim = F.cosine_similarity(enhanced_states, original_states, dim=-1)
        contrastive_loss = -torch.log(torch.exp(cos_sim) / torch.exp(cos_sim).sum())
        
        # Add regularization loss
        reg_loss = 0.01 * (enhanced_states.pow(2).mean() + original_states.pow(2).mean())
        
        return contrastive_loss.mean() + reg_loss

    def training_step(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Custom training step that ensures proper device placement and gradient scaling."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        # Move auxiliary modules to correct device if available
        device = inputs[list(inputs.keys())[0]].device
        for module in [self.memory_module, self.tree_module, self.cache_module]:
            if module is not None:
                module.train()
                module.to(device)

        # Enable automatic mixed precision training
        with autocast():
            loss = self.compute_loss(model, inputs)
            
        # Scale loss and backward pass
        self.scaler.scale(loss).backward()
        
        if self.steps % self.gradient_accumulation_steps == 0:
            # Unscale gradients and clip
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            # Optimizer step with scaler
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        
        self.steps += 1
        return loss.detach()

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Save model along with memory, tree, and cache components."""
        # Save main model
        super().save_model(output_dir, _internal_call)
        
        if output_dir is not None:
            # Save additional components
            if self.memory_module is not None:
                self.memory_module.save_pretrained(output_dir)
                
            if self.tree_module is not None:
                self.tree_module.save_pretrained(output_dir)
                
            if self.cache_module is not None:
                self.cache_module.save_pretrained(output_dir)
