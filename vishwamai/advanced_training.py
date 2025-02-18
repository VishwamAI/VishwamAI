import torch
import torch.nn as nn
from typing import Optional, Dict, Any, List, Tuple, Union
from dataclasses import dataclass, asdict
import math
import numpy as np
from .neural_memory import NeuralMemory
from .tree_of_thoughts import TreeConfig
from .curriculum import CurriculumConfig
from .config import ModelArgs

@dataclass
class TrainingStats:
    """Container for training statistics."""
    loss: float
    lr: float
    step: int
    memory_usage: Dict[str, float]
    curriculum_stats: Dict[str, Any]
    tot_stats: Dict[str, Any]
    memory_stats: Dict[str, Any]
    moe_metrics: Optional[Dict[str, float]] = None
    gradient_norm: Optional[float] = None
    eval_score: Optional[float] = None

class AdvancedTrainer:
    """Advanced training manager with curriculum learning, neural memory, and tree of thoughts."""
    
    def __init__(
        self,
        model: nn.Module,
        config: Union[Dict[str, Any], ModelArgs],
        device: torch.device,
        memory_size: int = 512,
        cache_size: int = 256,
        tot_config: Optional[TreeConfig] = None,
        reward_config: Optional[Dict] = None,
        curriculum_config: Optional[CurriculumConfig] = None,
        neural_memory: Optional[NeuralMemory] = None
    ):
        self.model = model
        self.device = device
        self.memory_size = memory_size  # Store memory_size as instance variable
        self.cache_size = cache_size
        
        # Convert ModelArgs to dict if necessary
        if isinstance(config, ModelArgs):
            self.config = asdict(config)
        else:
            self.config = config
            
        # Default training parameters
        defaults = {
            'early_stop_patience': 1000,
            'learning_rate': 1e-4,
            'min_learning_rate': 1e-5,
            'warmup_steps': 2000,
            'max_steps': 100000,
            'batch_size': 32,
            'accumulation_steps': 1,
            'weight_decay': 0.01,
            'gradient_clip': 1.0,
            'lr_decay_steps': 50000
        }
        
        # Initialize parameters with defaults
        for key, default in defaults.items():
            setattr(self, key, self.config.get(key, default))
        
        # Initialize components
        self.neural_memory = neural_memory
        self.tot_config = tot_config
        self.reward_config = reward_config
        self.curriculum_config = curriculum_config
        
        # Initialize training state
        self.current_step = 0
        self.memory_cache = {}
        self.best_loss = float('inf')
        self.steps_since_improve = 0
        
        # Learning rate scheduling
        self.base_lr = float(self.learning_rate)
        self.min_lr = float(self.min_learning_rate)
        
        # Initialize optimizer and other components
        self._initialize_components()

    def _initialize_components(self):
        """Initialize trainer components after configuration setup."""
        try:
            # Initialize optimizer
            self.optimizer = self._create_optimizer()
            self.scheduler = self._create_scheduler()
            
            # Initialize neural memory if not provided
            if self.neural_memory is None:
                model_dim = getattr(self.model, 'dim', None)
                if model_dim is None:
                    model_dim = self.config.get('dim', 2048)
                
                self.neural_memory = NeuralMemory(
                    memory_size=self.memory_size,
                    hidden_dim=model_dim,
                    num_memory_heads=4
                ).to(self.device)
            
            # Initialize tracking states
            self.current_difficulty = 0.0
            self.curriculum_scores = []
            self.thought_buffer = []
            self.best_thoughts = []
            self.eval_history = []
            self.loss_history = []
            
        except Exception as e:
            print(f"Error initializing components: {str(e)}")
            raise

    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer with parameter groups."""
        # Separate parameters for different optimization strategies
        decay_params = []
        no_decay_params = []
        moe_params = []
        
        for name, param in self.model.named_parameters():
            if 'bias' in name or 'layer_norm' in name:
                no_decay_params.append(param)
            elif 'moe' in name:
                moe_params.append(param)
            else:
                decay_params.append(param)
        
        param_groups = [
            {'params': decay_params, 'weight_decay': self.config.get('weight_decay', 0.1)},
            {'params': no_decay_params, 'weight_decay': 0.0},
            {'params': moe_params, 'weight_decay': self.config.get('moe_weight_decay', 0.05)}
        ]
        
        return torch.optim.AdamW(
            param_groups,
            lr=self.base_lr,
            betas=(0.9, 0.95),
            eps=1e-8
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler."""
        return torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            self._lr_lambda
        )
    
    def _lr_lambda(self, step: int) -> float:
        """Compute learning rate multiplier based on step."""
        if step < self.warmup_steps:
            return float(step) / float(max(1, self.warmup_steps))
        decay_ratio = float(step - self.warmup_steps) / float(max(1, self.lr_decay_steps))
        return max(self.min_lr / self.base_lr, 0.5 * (1.0 + math.cos(math.pi * decay_ratio)))

    def train_step(self, batch: Optional[Dict[str, torch.Tensor]] = None) -> TrainingStats:
        """Execute one training step."""
        self.model.train()
        stats = {}
        
        try:
            # Adjust difficulty based on curriculum
            if self.curriculum_config:
                self._adjust_curriculum_difficulty()
            
            # Forward pass through model
            if batch is not None:
                outputs = self.model(**batch)
            else:
                # Generate synthetic batch if none provided
                batch_size = self.config.get('batch_size', 4)
                seq_len = self.config.get('max_seq_len', 2048)
                dummy_input = torch.randint(
                    0, self.config.get('vocab_size', 32000),
                    (batch_size, seq_len),
                    device=self.device
                )
                outputs = self.model(dummy_input)
            
            # Apply Tree of Thoughts if configured
            if self.tot_config:
                outputs = self._apply_tree_of_thoughts(outputs)
            
            # Process with neural memory
            if self.neural_memory is not None:
                memory_output = self.neural_memory(outputs)
                loss = self.compute_loss(memory_output, batch)
            else:
                loss = self.compute_loss(outputs, batch)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('max_grad_norm', 1.0)
            )
            
            self.optimizer.step()
            self.scheduler.step()
            self.current_step += 1
            
            # Update best loss and check early stopping
            if loss.item() < self.best_loss:
                self.best_loss = loss.item()
                self.steps_since_improve = 0
            else:
                self.steps_since_improve += 1
            
            # Collect training statistics
            stats = TrainingStats(
                loss=loss.item(),
                lr=self.optimizer.param_groups[0]['lr'],
                step=self.current_step,
                memory_usage={
                    'allocated': torch.cuda.memory_allocated(self.device) / 1024**2,
                    'cached': torch.cuda.memory_reserved(self.device) / 1024**2
                },
                curriculum_stats=self._get_curriculum_stats(),
                tot_stats=self._get_tot_stats(),
                memory_stats=self._get_memory_stats(),
                moe_metrics=self._get_moe_metrics() if hasattr(self.model, 'moe') else None,
                gradient_norm=grad_norm.item()
            )
            
            # Update tracking histories
            self.loss_history.append(loss.item())
            
            # Update neural memory cache
            if len(self.memory_cache) > self.cache_size:
                self.memory_cache.clear()
            
            # Check for early stopping
            if self.steps_since_improve > self.early_stop_patience:
                print("Early stopping triggered!")
                raise StopIteration("Early stopping triggered")
            
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            raise
            
        return stats
    
    def _adjust_curriculum_difficulty(self):
        """Adjust curriculum difficulty based on recent performance."""
        if len(self.curriculum_scores) >= 50:  # Need enough samples
            avg_score = np.mean(self.curriculum_scores[-50:])
            if avg_score > 0.8:  # Performance threshold
                self.current_difficulty = min(1.0, self.current_difficulty + 0.1)
            elif avg_score < 0.6:
                self.current_difficulty = max(0.0, self.current_difficulty - 0.05)
    
    def _apply_tree_of_thoughts(self, outputs: torch.Tensor) -> torch.Tensor:
        """Apply Tree of Thoughts reasoning."""
        if not self.tot_config:
            return outputs
            
        # Generate multiple reasoning paths
        thoughts = []
        for _ in range(self.tot_config.max_branches):
            thought = self.model.generate(
                outputs,
                max_length=self.tot_config.max_depth,
                num_beams=self.tot_config.beam_width
            )
            thoughts.append(thought)
        
        # Evaluate thoughts and select best
        scores = self._evaluate_thoughts(thoughts)
        best_idx = torch.argmax(scores)
        
        # Store best thought
        self.best_thoughts.append(thoughts[best_idx])
        
        return thoughts[best_idx]
    
    def _evaluate_thoughts(self, thoughts: List[torch.Tensor]) -> torch.Tensor:
        """Evaluate quality of different thoughts."""
        scores = []
        for thought in thoughts:
            # Compute metrics
            coherence = self._compute_coherence(thought)
            relevance = self._compute_relevance(thought)
            novelty = self._compute_novelty(thought)
            
            # Combine scores
            score = (coherence + relevance + novelty) / 3
            scores.append(score)
        
        return torch.tensor(scores, device=self.device)
    
    def compute_loss(self, outputs: torch.Tensor, batch: Optional[Dict[str, torch.Tensor]]) -> torch.Tensor:
        """Compute training loss."""
        if batch is not None and 'labels' in batch:
            return torch.nn.functional.cross_entropy(
                outputs.view(-1, outputs.size(-1)),
                batch['labels'].view(-1)
            )
        return outputs.mean()  # Dummy loss for testing
    
    def save_checkpoint(self, path: str):
        """Save training checkpoint."""
        torch.save({
            'step': self.current_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'neural_memory_state': self.neural_memory.state_dict() if self.neural_memory else None,
            'curriculum_difficulty': self.current_difficulty,
            'best_loss': self.best_loss,
            'steps_since_improve': self.steps_since_improve,
            'loss_history': self.loss_history,
            'eval_history': self.eval_history,
            'config': self.config
        }, path)
    
    def load_state_dict(self, checkpoint: Dict[str, Any]):
        """Load trainer state from checkpoint."""
        self.current_step = checkpoint.get('step', 0)
        self.optimizer.load_state_dict(checkpoint.get('optimizer_state_dict', {}))
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'neural_memory_state' in checkpoint and self.neural_memory:
            self.neural_memory.load_state_dict(checkpoint['neural_memory_state'])
        self.current_difficulty = checkpoint.get('curriculum_difficulty', 0.0)
        self.best_loss = checkpoint.get('best_loss', float('inf'))
        self.steps_since_improve = checkpoint.get('steps_since_improve', 0)
        self.loss_history = checkpoint.get('loss_history', [])
        self.eval_history = checkpoint.get('eval_history', [])
    
    def evaluate(self, eval_data: Optional[Dict[str, torch.Tensor]] = None) -> Dict[str, float]:
        """Run model evaluation."""
        self.model.eval()
        stats = {}
        
        with torch.no_grad():
            if eval_data is not None:
                outputs = self.model(**eval_data)
                loss = self.compute_loss(outputs, eval_data)
                stats['eval_loss'] = loss.item()
                
                # Compute additional metrics
                if hasattr(self.model, 'compute_perplexity'):
                    stats['perplexity'] = self.model.compute_perplexity(outputs).item()
                
                # Store evaluation result
                self.eval_history.append((self.current_step, stats))
        
        return stats
    
    def _get_curriculum_stats(self) -> Dict[str, Any]:
        """Get curriculum learning statistics."""
        return {
            'current_difficulty': self.current_difficulty,
            'recent_scores': self.curriculum_scores[-10:] if self.curriculum_scores else [],
            'progress': self.current_difficulty / 1.0  # Progress from 0 to 1
        }
    
    def _get_tot_stats(self) -> Dict[str, Any]:
        """Get Tree of Thoughts statistics."""
        return {
            'num_thoughts': len(self.thought_buffer),
            'best_thoughts': len(self.best_thoughts),
            'thought_quality': np.mean([t.score for t in self.best_thoughts]) if self.best_thoughts else 0
        }
    
    def _get_memory_stats(self) -> Dict[str, Any]:
        """Get neural memory statistics."""
        if not self.neural_memory:
            return {}
        return {
            'memory_size': self.neural_memory.memory_size,
            'cache_size': len(self.memory_cache),
            'usage': (torch.sum(self.neural_memory.usage_tracker > 0).item() / 
                     self.neural_memory.memory_size)
        }
    
    def _get_moe_metrics(self) -> Dict[str, float]:
        """Get Mixture of Experts metrics if available."""
        if not hasattr(self.model, 'moe'):
            return {}
        return {
            'expert_utilization': self.model.moe.get_expert_utilization(),
            'load_balancing_loss': self.model.moe.get_load_balancing_loss(),
            'expert_capacity': self.model.moe.get_expert_capacity()
        }
    
    def _compute_coherence(self, thought: torch.Tensor) -> float:
        """Compute coherence score for a thought."""
        # Implement coherence metric
        return torch.rand(1).item()  # Placeholder
    
    def _compute_relevance(self, thought: torch.Tensor) -> float:
        """Compute relevance score for a thought."""
        # Implement relevance metric
        return torch.rand(1).item()  # Placeholder
    
    def _compute_novelty(self, thought: torch.Tensor) -> float:
        """Compute novelty score for a thought."""
        # Implement novelty metric
        return torch.rand(1).item()  # Placeholder
