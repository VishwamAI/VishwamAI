"""
Training and optimization configuration
"""
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class T4TrainingConfig:
    """T4-specific training configuration"""
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    warmup_steps: int = 1000
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    mixed_precision: str = "fp16"
    gradient_checkpointing: bool = True
    
@dataclass
class TrainingConfig:
    """
    Training configuration parameters
    """
    # Basic training params
    batch_size: int = 64
    num_epochs: int = 10
    learning_rate: float = 1e-4
    warmup_steps: int = 2000
    max_steps: Optional[int] = None
    
    # Optimization
    optimizer: str = "adamw"  # adamw, adam, sgd
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    gradient_accumulation_steps: int = 1
    
    # Learning rate schedule
    lr_scheduler: str = "linear"  # linear, cosine, constant
    lr_warmup_ratio: float = 0.1
    min_lr_ratio: float = 0.1
    
    # Mixed precision training
    mixed_precision: Optional[str] = None  # None, 'fp16', 'bf16'
    loss_scaling: str = "dynamic"  # dynamic, static
    initial_scale: float = 2**16
    
    # Memory optimization
    gradient_checkpointing: bool = False
    zero_stage: int = 0  # 0, 1, 2, 3
    
    # Logging and saving
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    
    # Data loading
    num_workers: int = 4
    pin_memory: bool = True
    
    # Tree planner specific
    planning_loss_weight: float = 1.0
    value_loss_weight: float = 0.5
    entropy_weight: float = 0.01
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'TrainingConfig':
        """Create config from dictionary"""
        return cls(**config_dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return {
            k: v for k, v in self.__dict__.items()
            if not k.startswith('_')
        }
        
    @classmethod
    def get_t4_config(cls) -> 'TrainingConfig':
        """Get T4-optimized training configuration"""
        t4_config = T4TrainingConfig()
        return cls(
            batch_size=t4_config.batch_size,
            gradient_accumulation_steps=t4_config.gradient_accumulation_steps,
            learning_rate=t4_config.learning_rate,
            warmup_steps=t4_config.warmup_steps,
            weight_decay=t4_config.weight_decay,
            max_grad_norm=t4_config.max_grad_norm,
            mixed_precision=t4_config.mixed_precision,
            gradient_checkpointing=t4_config.gradient_checkpointing,
            optimizer="adamw",
            lr_scheduler="linear",
            zero_stage=1  # Enable basic ZeRO optimization
        )

    def get_optimizer_params(self, model) -> List[Dict[str, Any]]:
        """
        Get layerwise learning rates and weight decay
        
        Args:
            model: The model to get parameters for
            
        Returns:
            List of parameter groups with optimization settings
        """
        # Layer-wise learning rate decay
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_params = [
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [
                    p for n, p in model.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        return optimizer_grouped_params
