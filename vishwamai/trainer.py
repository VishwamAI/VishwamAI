import torch
from transformers import PreTrainedModel, Trainer, TrainingArguments
from typing import Dict, Union, Any, Optional
from .neural_memory import ReasoningMemoryTransformer
from .tree_of_thoughts import TreeOfThoughts
from .cache_augmentation import DifferentiableCacheAugmentation

class VishwamAIPretrainer(Trainer):
    def __init__(self, 
                 memory_module: Optional[ReasoningMemoryTransformer] = None,
                 tree_module: Optional[TreeOfThoughts] = None,
                 cache_module: Optional[DifferentiableCacheAugmentation] = None,
                 **kwargs):
        super().__init__(**kwargs)
        
        # Initialize memory tracking 
        self.memory_module = memory_module
        self.tree_module = tree_module
        self.cache_module = cache_module

    def compute_loss(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor], return_outputs=False):
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

        if return_outputs:
            return total_loss, outputs
        return total_loss

    def compute_auxiliary_loss(self, enhanced_states: torch.Tensor, original_states: torch.Tensor) -> torch.Tensor:
        """Compute auxiliary loss between enhanced and original states."""
        # Cosine similarity loss
        enhanced_norm = torch.nn.functional.normalize(enhanced_states, p=2, dim=-1)
        original_norm = torch.nn.functional.normalize(original_states, p=2, dim=-1)
        cosine_sim = (enhanced_norm * original_norm).sum(-1).mean()
        
        # MSE loss with small weight
        mse_loss = torch.nn.functional.mse_loss(enhanced_states, original_states)
        
        # Combine losses with weights
        return 0.1 * (1.0 - cosine_sim) + 0.01 * mse_loss

    def training_step(self, model: PreTrainedModel, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Custom training step that ensures proper device placement."""
        model.train()
        inputs = self._prepare_inputs(inputs)

        if self.memory_module is not None:
            self.memory_module.train()
            self.memory_module.to(inputs[list(inputs.keys())[0]].device)
            
        if self.tree_module is not None:
            self.tree_module.train()
            self.tree_module.to(inputs[list(inputs.keys())[0]].device)
            
        if self.cache_module is not None:
            self.cache_module.train()
            self.cache_module.to(inputs[list(inputs.keys())[0]].device)

        loss = self.compute_loss(model, inputs)

        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        loss.backward()
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
