import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from .model_factory import ModelFactory, AdvancedModelConfig
from .curriculum import CurriculumConfig, CurriculumScheduler
from .hardware_adapter import HardwareConfig
from contextlib import nullcontext

@dataclass
class UnifiedTrainerConfig:
    """Configuration for unified advanced trainer."""
    model_config: AdvancedModelConfig
    curriculum_config: Optional[CurriculumConfig] = None
    hardware_config: Optional[HardwareConfig] = None
    mixed_precision: bool = True
    gradient_checkpointing: bool = True
    gradient_accumulation_steps: int = 1
    logging_steps: int = 100
    eval_steps: int = 1000
    save_steps: int = 5000
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000

class UnifiedTrainer:
    """
    Unified trainer integrating all advanced components.
    
    This trainer combines:
    - Curriculum learning
    - Emergent behavior
    - Integrated information processing
    - Ethical framework
    - Hardware optimization
    - Open-ended learning
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: UnifiedTrainerConfig,
        device: torch.device
    ):
        self.model = model
        self.config = config
        self.device = device
        
        # Initialize components
        self.components = ModelFactory.create_advanced_components(
            model,
            config.model_config
        )
        ModelFactory.initialize_components(self.components, device)
        
        # Verify component compatibility
        if not ModelFactory.verify_compatibility(self.components):
            raise ValueError("Component dimensions are incompatible")
        
        # Initialize curriculum if configured
        self.curriculum_scheduler = (
            CurriculumScheduler(config.curriculum_config)
            if config.curriculum_config else None
        )
        
        # Initialize training state
        self.global_step = 0
        self.epoch = 0
        self.best_performance = float('inf')
        self.training_history = []
        
        # Initialize mixed precision
        self.scaler = torch.cuda.amp.GradScaler() if config.mixed_precision else None
        
        # Initialize optimizers
        self._init_optimizers()
        
    def _init_optimizers(self):
        """Initialize optimizers for model and components."""
        # Collect all trainable parameters
        param_groups = []
        
        # Model parameters
        model_params = {'params': self.model.parameters(), 'name': 'model'}
        param_groups.append(model_params)
        
        # Component parameters
        for name, component in self.components.items():
            if hasattr(component, 'parameters'):
                component_params = {
                    'params': component.parameters(),
                    'name': name,
                    'lr': self.config.model_config.hidden_dim ** -0.5
                }
                param_groups.append(component_params)
                
        # Create optimizer
        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.model_config.hidden_dim ** -0.5,
            betas=(0.9, 0.98),
            eps=1e-9
        )
        
    def train_step(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """Execute single training step with all components."""
        try:
            self.model.train()
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Get curriculum parameters if enabled
            if self.curriculum_scheduler is not None:
                difficulty_score, _ = self.curriculum_scheduler.estimate_task_difficulty(
                    batch['input_ids'], batch['attention_mask']
                )
            else:
                difficulty_score = 0.0
                
            # Process with emergent behavior
            emergent_state, emergent_metrics = self.components['emergent'](
                batch['input_ids'],
                batch.get('labels')
            )
            
            # Process with integrated information
            integrated_state, integration_metrics = self.components['integration'](
                emergent_state
            )
            
            # Check ethical compliance
            ethical_state, ethical_metrics = self.components['ethical'](
                integrated_state,
                context=batch.get('attention_mask')
            )
            
            # Generate new learning targets
            open_ended_state, evolution_metrics = self.components['open_ended'](
                ethical_state,
                self.training_history[-100:] if self.training_history else None
            )
            
            # Forward pass with hardware optimization
            with torch.cuda.amp.autocast() if self.config.mixed_precision else nullcontext():
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch.get('labels'),
                    output_hidden_states=True
                )
                
                # Calculate losses
                main_loss = outputs.loss
                
                # Component-specific losses
                emergent_loss = torch.mean(torch.stack(
                    [m['loss'] for m in emergent_metrics.values() if 'loss' in m]
                )) if emergent_metrics else torch.tensor(0.0)
                
                integration_loss = torch.mean(torch.stack(
                    [m['loss'] for m in integration_metrics.values() if 'loss' in m]
                )) if integration_metrics else torch.tensor(0.0)
                
                ethical_loss = torch.mean(torch.stack(
                    [m['loss'] for m in ethical_metrics.values() if 'loss' in m]
                )) if ethical_metrics else torch.tensor(0.0)
                
                evolution_loss = torch.mean(torch.stack(
                    [m['loss'] for m in evolution_metrics.values() if 'loss' in m]
                )) if evolution_metrics else torch.tensor(0.0)
                
                # Combined loss
                total_loss = (
                    main_loss +
                    0.1 * emergent_loss +
                    0.1 * integration_loss +
                    0.2 * ethical_loss +
                    0.1 * evolution_loss
                )
                
            # Backward pass
            if self.config.mixed_precision:
                self.scaler.scale(total_loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )
                self.optimizer.step()
                
            self.optimizer.zero_grad()
            
            # Update curriculum if enabled
            if self.curriculum_scheduler is not None:
                self.curriculum_scheduler.update({
                    'loss': total_loss.item(),
                    'accuracy': (outputs.logits.argmax(-1) == batch['labels']).float().mean().item()
                    if 'labels' in batch else 0.0
                })
                
            # Collect metrics
            metrics = {
                'loss': total_loss.item(),
                'main_loss': main_loss.item(),
                'emergent_loss': emergent_loss.item(),
                'integration_loss': integration_loss.item(),
                'ethical_loss': ethical_loss.item(),
                'evolution_loss': evolution_loss.item(),
                **emergent_metrics,
                **integration_metrics,
                **ethical_metrics,
                **evolution_metrics
            }
            
            # Update history
            self.training_history.append(metrics)
            self.global_step += 1
            
            return metrics
            
        except Exception as e:
            print(f"Error in training step: {str(e)}")
            raise
            
    def evaluate(
        self,
        eval_dataloader: torch.utils.data.DataLoader
    ) -> Dict[str, float]:
        """Evaluate model with all components."""
        self.model.eval()
        total_loss = 0
        component_metrics = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass through components
                emergent_state, _ = self.components['emergent'](
                    batch['input_ids'],
                    batch.get('labels')
                )
                
                integrated_state, _ = self.components['integration'](emergent_state)
                ethical_state, _ = self.components['ethical'](integrated_state)
                
                # Model forward pass
                outputs = self.model(
                    input_ids=batch['input_ids'],
                    attention_mask=batch['attention_mask'],
                    labels=batch.get('labels')
                )
                
                total_loss += outputs.loss.item()
                
                # Collect component metrics
                component_metrics.append(
                    ModelFactory.get_component_stats(self.components)
                )
                
        # Average metrics
        avg_loss = total_loss / len(eval_dataloader)
        avg_component_metrics = {}
        
        for metrics in component_metrics:
            for k, v in metrics.items():
                if k not in avg_component_metrics:
                    avg_component_metrics[k] = []
                avg_component_metrics[k].append(v)
                
        avg_component_metrics = {
            k: sum(v) / len(v) for k, v in avg_component_metrics.items()
        }
        
        return {
            'eval_loss': avg_loss,
            **avg_component_metrics
        }
        
    def save_checkpoint(self, path: str):
        """Save unified training checkpoint."""
        checkpoint = {
            'model_state': self.model.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'components': {
                name: component.state_dict()
                for name, component in self.components.items()
            },
            'curriculum_state': (
                self.curriculum_scheduler.state_dict()
                if self.curriculum_scheduler else None
            ),
            'training_state': {
                'global_step': self.global_step,
                'epoch': self.epoch,
                'best_performance': self.best_performance,
                'training_history': self.training_history
            },
            'config': self.config
        }
        
        torch.save(checkpoint, path)
        
    def load_checkpoint(self, path: str):
        """Load unified training checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        
        for name, state in checkpoint['components'].items():
            self.components[name].load_state_dict(state)
            
        if checkpoint['curriculum_state'] and self.curriculum_scheduler:
            self.curriculum_scheduler.load_state_dict(
                checkpoint['curriculum_state']
            )
            
        # Restore training state
        training_state = checkpoint['training_state']
        self.global_step = training_state['global_step']
        self.epoch = training_state['epoch']
        self.best_performance = training_state['best_performance']
        self.training_history = training_state['training_history']
