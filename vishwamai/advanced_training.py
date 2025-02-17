import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, List, Dict, Any
import math
import numpy as np
from contextlib import nullcontext
from .model_utils import create_attention_mask
from .neural_memory import NeuralMemory
from .cache_augmentation import CacheAugmentation
from .MoE import MoEConfig, create_moe_layer
from .tree_of_thoughts import TreeOfThoughts, TreeConfig, RewardConfig

class AdvancedTrainer:
    """Advanced trainer implementing MoE, neural memory, and cache augmentation."""
    
    def __init__(
        self,
        model: nn.Module,
        config: dict,
        device: torch.device,
        memory_size: int = 1024,
        cache_size: int = 512,
        tot_config: Optional[TreeConfig] = None,
        reward_config: Optional[RewardConfig] = None
    ):
        """
        Initialize the advanced trainer.
        
        Args:
            model: The model to train
            config: Training configuration
            device: Device to train on
            memory_size: Size of neural memory
            cache_size: Size of cache
        """
        self.model = model
        self.config = config
        self.device = device
        
        # Initialize Tree of Thoughts
        self.tot_config = tot_config or TreeConfig()
        self.reward_config = reward_config or RewardConfig()
        self.tot = TreeOfThoughts(
            model=model,
            config=self.tot_config,
            reward_config=self.reward_config
        ).to(device)
        
        # Initialize memory and cache with hierarchical structure
        self.memory = NeuralMemory(
            memory_size=memory_size,
            hidden_dim=config.get('hidden_dim', 768),
            num_heads=config.get('num_heads', 8),
            sparsity=config.get('memory_sparsity', 0.9)
        ).to(device)
        
        self.cache = CacheAugmentation(
            cache_size=cache_size,
            hidden_dim=config.get('hidden_dim', 768),
            num_heads=config.get('num_heads', 8)
        ).to(device)
        
        # Advanced training parameters
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 1)
        self.mixed_precision = config.get('mixed_precision', True)
        self.gradient_checkpointing = config.get('gradient_checkpointing', True)
        self.tree_of_thoughts_depth = config.get('tree_of_thoughts_depth', 3)
        
        # MoE specific parameters
        self.moe_aux_loss_weight = config.get('moe_aux_loss_weight', 0.01)
        self.expert_load_balance_weight = config.get('expert_load_balance_weight', 0.01)
        
        # Training state tracking
        self.training_steps = 0
        self.loss_history = []
        self.expert_usage_history = []
        
        # Dynamic batch sizing
        self._init_batch_sizing()
        
        # Initialize training components
        self._init_optimizers()
        self._init_schedulers()
        
        # Initialize gradient scaler for mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if self.mixed_precision else None

    def _init_batch_sizing(self):
        """Initialize dynamic batch sizing parameters."""
        self.initial_batch_size = self.config.get('batch_size', 32)
        self.current_batch_size = self.initial_batch_size
        self.batch_size_update_freq = 100
        self.last_batch_size_update = 0
        self.min_batch_size = self.config.get('min_batch_size', 16)
        self.max_batch_size = self.config.get('max_batch_size', 128)
        self.batch_size_history = []
        self.loss_window = []
        self.window_size = 50

    def _init_optimizers(self):
        """Initialize optimizers with MoE-aware parameter groups."""
        # Separate MoE and non-MoE parameters
        param_groups = self._create_parameter_groups()
        
        # AdaFactor with dynamic scaling and gradient clipping
        from transformers.optimization import Adafactor
        self.optimizer = Adafactor(
            param_groups,
            scale_parameter=True,
            relative_step=True,
            warmup_init=True,
            lr=None,
            clip_threshold=self.config.get('max_grad_norm', 1.0),
            decay_rate=self.config.get('decay_rate', -0.8),
            beta1=self.config.get('beta1', None),
            weight_decay=self.config.get('weight_decay', 0.01)
        )
        
        # Secondary optimizer for memory components
        self.memory_optimizer = torch.optim.Adam(
            self.memory.parameters(),
            lr=self.config.get('memory_lr', 1e-4)
        )

    def _create_parameter_groups(self) -> List[Dict[str, Any]]:
        """Create parameter groups for optimization."""
        moe_params = []
        base_params = []
        
        for name, param in self.model.named_parameters():
            if 'moe' in name:
                moe_params.append(param)
            else:
                base_params.append(param)
        
        return [
            {'params': base_params},
            {'params': moe_params, 'lr_scale': 0.1}  # Lower LR for MoE params
        ]
    
    def _init_schedulers(self):
        """Initialize learning rate schedulers."""
        from transformers import get_cosine_schedule_with_warmup
        
        num_training_steps = self.config.get('num_training_steps', 100000)
        num_warmup_steps = self.config.get('num_warmup_steps', 5000)
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps
        )
    
    def _collect_expert_metrics(self) -> Dict:
        """
        Collect MoE expert usage and balance metrics.
        
        Returns:
            Dict containing expert usage statistics and load balancing metrics
        """
        expert_metrics = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'gate') and hasattr(module.gate, 'expert_counts'):
                expert_counts = module.gate.expert_counts.detach().cpu()
                total_tokens = expert_counts.sum().item()
                expert_usage = expert_counts / total_tokens
                
                # Calculate load balancing metrics
                cv = torch.std(expert_usage) / torch.mean(expert_usage)
                
                expert_metrics[f'{name}_metrics'] = {
                    'usage_std': torch.std(expert_usage).item(),
                    'usage_cv': cv.item(),
                    'max_usage': torch.max(expert_usage).item(),
                    'min_usage': torch.min(expert_usage).item(),
                    'expert_counts': expert_counts.tolist()
                }
        
        return expert_metrics

    def _explore_thoughts(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> List[torch.Tensor]:
        """
        Implement tree of thoughts exploration.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            
        Returns:
            List of thought tensors
        """
        thoughts = []
        current_thought = input_ids
        
        for depth in range(self.tree_of_thoughts_depth):
            # Generate next thought level
            with torch.no_grad():
                outputs = self.model(
                    input_ids=current_thought,
                    attention_mask=attention_mask,
                    output_hidden_states=True
                )
                next_thought = outputs.hidden_states[-1]
                thoughts.append(next_thought)
                current_thought = next_thought
                
        return thoughts

    def _augment_with_memory(
        self,
        input_ids: torch.Tensor,
        memory_output: torch.Tensor
    ) -> torch.Tensor:
        """
        Augment input with neural memory.
        
        Args:
            input_ids: Input token IDs
            memory_output: Output from neural memory
            
        Returns:
            Augmented input tensor
        """
        # Combine input embeddings with memory output
        input_embeds = self.model.get_input_embeddings()(input_ids)
        augmented = input_embeds + memory_output
        return augmented

    def _integrate_cache_knowledge(
        self,
        input_tensor: torch.Tensor,
        cache_info: torch.Tensor
    ) -> torch.Tensor:
        """
        Integrate cache knowledge with input.
        
        Args:
            input_tensor: Input tensor
            cache_info: Cache information
            
        Returns:
            Knowledge-integrated tensor
        """
        return input_tensor + self.cache.integrate_knowledge(cache_info)

    def _calculate_memory_loss(
        self,
        memory_output: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Calculate memory-based loss component."""
        return F.mse_loss(memory_output, hidden_states.detach())

    def _calculate_cache_loss(
        self,
        cache_info: torch.Tensor,
        hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """Calculate cache-based loss component."""
        return self.cache.compute_loss(cache_info, hidden_states)

    def _calculate_thought_loss(
        self,
        thoughts: List[torch.Tensor],
        logits: torch.Tensor
    ) -> torch.Tensor:
        """Calculate thought-based loss component."""
        thought_loss = torch.tensor(0.0, device=logits.device)
        for thought in thoughts:
            thought_loss += F.kl_div(
                F.log_softmax(thought, dim=-1),
                F.softmax(logits.detach(), dim=-1),
                reduction='batchmean'
            )
        return thought_loss / len(thoughts)

    def _update_batch_size(self, current_loss: float):
        """
        Update batch size based on training dynamics.
        
        Args:
            current_loss: Current training loss
        """
        self.loss_window.append(current_loss)
        if len(self.loss_window) > self.window_size:
            self.loss_window.pop(0)
        
        if self.training_steps - self.last_batch_size_update >= self.batch_size_update_freq:
            if len(self.loss_window) >= self.window_size:
                loss_std = np.std(self.loss_window)
                loss_mean = np.mean(self.loss_window)
                cv = loss_std / loss_mean
                
                # Adjust batch size based on loss stability
                if cv < 0.1:  # Stable loss
                    self.current_batch_size = min(
                        self.current_batch_size * 1.1,
                        self.max_batch_size
                    )
                elif cv > 0.2:  # Unstable loss
                    self.current_batch_size = max(
                        self.current_batch_size * 0.9,
                        self.min_batch_size
                    )
                
                self.batch_size_history.append(self.current_batch_size)
                self.last_batch_size_update = self.training_steps

    def _process_with_tot(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor
    ) -> Tuple[torch.Tensor, List[Dict]]:
        """Process hidden states through Tree of Thoughts reasoning."""
        outputs = self.tot(hidden_states)
        best_nodes = self.tot._get_leaf_nodes(outputs)
        if best_nodes:
            best_node = max(best_nodes, key=lambda x: x.score)
            return best_node.state, best_node.reasoning_steps
        return hidden_states, []

    def deep_calculation_step(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform deep calculation step with MoE support.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            labels: Optional labels
            
        Returns:
            Tuple of (total_loss, logits)
        """
        # Tree of thoughts exploration
        thoughts = self._explore_thoughts(input_ids, attention_mask)
        
        # Neural memory augmentation
        memory_output = self.memory(input_ids)
        augmented_input = self._augment_with_memory(input_ids, memory_output)
        
        # Get initial hidden states
        with torch.no_grad():
            initial_hidden = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            ).hidden_states[-1]
        
        # Process through Tree of Thoughts
        tot_hidden, reasoning_steps = self._process_with_tot(initial_hidden, attention_mask)
        
        # Cache-based knowledge integration
        cache_info = self.cache.retrieve(input_ids)
        enriched_input = self._integrate_cache_knowledge(tot_hidden, cache_info)
        
        try:
            # Forward pass with advanced attention
            outputs = self.model(
                input_ids=enriched_input,
                attention_mask=attention_mask,
                labels=labels,
                output_attentions=True,
                output_hidden_states=True
            )
            
            # Loss calculation
            base_loss = outputs.loss
            memory_loss = self._calculate_memory_loss(memory_output, outputs.hidden_states[-1])
            cache_loss = self._calculate_cache_loss(cache_info, outputs.hidden_states[-1])
            thought_loss = self._calculate_thought_loss(thoughts, outputs.logits)
            
            # Tree of Thoughts loss
            tot_loss = torch.tensor(0.0, device=self.device)
            if reasoning_steps:
                tot_rewards = [self.tot.reward_function(step['state'], step['operation']) 
                             for step in reasoning_steps]
                tot_loss = -torch.mean(torch.stack(tot_rewards))
            
            # MoE auxiliary loss
            moe_loss = self._calculate_moe_loss()
            
            # Combined loss
            total_loss = (
                base_loss +
                memory_loss +
                cache_loss +
                thought_loss +
                moe_loss +
                tot_loss * self.tot_config.reward_gamma
            )
            
            return total_loss, outputs.logits
            
        except RuntimeError as e:
            print(f"Error in deep calculation step: {str(e)}")
            raise

    def _calculate_moe_loss(self) -> torch.Tensor:
        """Calculate MoE-specific losses."""
        moe_loss = torch.tensor(0.0, device=self.device)
        
        if hasattr(self.model, 'moe_loss') and self.model.moe_loss is not None:
            moe_loss = self.model.moe_loss * self.moe_aux_loss_weight
            
            # Add expert load balancing loss
            expert_metrics = self._collect_expert_metrics()
            if expert_metrics:
                avg_cv = np.mean([m['usage_cv'] for m in expert_metrics.values()])
                load_balance_loss = avg_cv * self.expert_load_balance_weight
                moe_loss += load_balance_loss
                
        return moe_loss

    def train_step(
        self,
        batch: Dict[str, torch.Tensor],
        gradient_accumulation: bool = True
    ) -> Dict[str, Any]:
        """
        Execute single training step with MoE support.
        
        Args:
            batch: Training batch
            gradient_accumulation: Whether to use gradient accumulation
            
        Returns:
            Dict containing training statistics
        """
        try:
            # Get inputs
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch.get('labels')
            if labels is not None:
                labels = labels.to(self.device)
                
            # Enable gradient checkpointing if configured
            if self.gradient_checkpointing:
                self.model.gradient_checkpointing_enable()
                
            # Clear gradients
            self.optimizer.zero_grad()
            self.memory_optimizer.zero_grad()
            
            # Forward pass with deep calculation and memory management
            with torch.cuda.amp.autocast() if self.mixed_precision else nullcontext():
                loss, logits = self.deep_calculation_step(
                    input_ids,
                    attention_mask,
                    labels
                )
            
            # Handle loss scaling and backward pass
            self._handle_backward_pass(loss, gradient_accumulation)
            
            # Update training state
            self.training_steps += 1
            self._update_batch_size(loss.item())
            
            # Collect metrics
            stats = self._collect_training_stats(loss, logits)
            
            return stats
            
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            raise

    def _handle_backward_pass(self, loss: torch.Tensor, gradient_accumulation: bool):
        """Handle backward pass with mixed precision support."""
        if gradient_accumulation:
            loss = loss / self.gradient_accumulation_steps
            
        if self.mixed_precision:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            self._clip_gradients()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self._clip_gradients()
            self.optimizer.step()
            
        self.memory_optimizer.step()
        self.scheduler.step()

    def _clip_gradients(self):
        """Apply gradient clipping."""
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.get('max_grad_norm', 1.0)
        )

    def _collect_training_stats(
        self,
        loss: torch.Tensor,
        logits: torch.Tensor
    ) -> Dict[str, Any]:
        """Collect comprehensive training statistics."""
        expert_metrics = self._collect_expert_metrics()
        self.expert_usage_history.append(expert_metrics)
        
        stats = {
            'loss': loss.item(),
            'logits': logits,
            'lr': self.scheduler.get_last_lr()[0],
            'batch_size': self.current_batch_size,
            'memory_usage': self._get_memory_usage(),
            'moe_metrics': expert_metrics,
            'memory_stats': self.memory.get_memory_state(),
            'cache_stats': self.cache.get_cache_stats(),
            'training_step': self.training_steps,
            'gradient_norm': self._get_gradient_norm(),
            'tot_stats': {
                'num_nodes': len(list(self.tot._get_leaf_nodes([]))),
                'avg_depth': sum(n.depth for n in self.tot._get_leaf_nodes([])) / 
                           max(1, len(list(self.tot._get_leaf_nodes([])))),
                'avg_uncertainty': sum(n.uncertainty for n in self.tot._get_leaf_nodes([])) /
                                max(1, len(list(self.tot._get_leaf_nodes([]))))
            }
        }
        
        if hasattr(self.model, 'moe_loss'):
            stats['moe_loss'] = self.model.moe_loss.item()
            
        return stats

    def _get_memory_usage(self) -> Dict[str, float]:
        """Get current GPU memory usage."""
        return {
            'allocated': torch.cuda.memory_allocated(self.device) / 1024**2,  # MB
            'cached': torch.cuda.memory_reserved(self.device) / 1024**2,  # MB
            'max_allocated': torch.cuda.max_memory_allocated(self.device) / 1024**2  # MB
        }

    def _get_gradient_norm(self) -> float:
        """Calculate total gradient norm."""
        total_norm = 0.0
        for p in self.model.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        return math.sqrt(total_norm)

    def save_checkpoint(self, path: str):
        """
        Save training checkpoint with MoE state.
        
        Args:
            path: Path to save checkpoint
        """
        try:
            checkpoint = {
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'memory_state': self.memory.state_dict(),
                'cache_state': self.cache.state_dict(),
                'tot_state': self.tot.state_dict(),
                'scaler_state': self.scaler.state_dict() if self.mixed_precision else None,
                'config': self.config,
                'training_state': {
                    'steps': self.training_steps,
                    'loss_history': self.loss_history,
                    'expert_usage_history': self.expert_usage_history,
                    'batch_size_history': self.batch_size_history,
                    'current_batch_size': self.current_batch_size
                }
            }
            torch.save(checkpoint, path)
        except Exception as e:
            print(f"Error saving checkpoint: {str(e)}")
            raise

    def load_checkpoint(self, path: str):
        """
        Load training checkpoint with MoE state.
        
        Args:
            path: Path to load checkpoint from
        """
        try:
            checkpoint = torch.load(path, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            self.memory.load_state_dict(checkpoint['memory_state'])
            self.cache.load_state_dict(checkpoint['cache_state'])
            self.tot.load_state_dict(checkpoint['tot_state'])
            
            if self.mixed_precision and checkpoint['scaler_state']:
                self.scaler.load_state_dict(checkpoint['scaler_state'])
                
            self.config.update(checkpoint['config'])
            
            # Restore training state
            training_state = checkpoint.get('training_state', {})
            self.training_steps = training_state.get('steps', 0)
            self.loss_history = training_state.get('loss_history', [])
            self.expert_usage_history = training_state.get('expert_usage_history', [])
            self.batch_size_history = training_state.get('batch_size_history', [])
            self.current_batch_size = training_state.get('current_batch_size', self.initial_batch_size)
            
        except Exception as e:
            print(f"Error loading checkpoint: {str(e)}")
            raise
