import torch
from torch.utils.data import DataLoader, Dataset
import argparse
from pathlib import Path
from transformers import AutoModelForCausalLM, TrainingArguments
from tqdm import tqdm
from typing import Dict, List, Any, Optional
import json
import logging
import wandb
from torch.cuda.amp import autocast
import numpy as np
from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional
from ..trainer import VishwamAIPretrainer
from vishwamai.neural_memory import NeuralMemory
from vishwamai.cache_augmentation import CacheAugmentation
from collections import defaultdict
from vishwamai.tree_of_thoughts import TreeOfThoughts, TreeConfig, RewardConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    difficulty_levels: List[str] = None
    steps_per_level: int = 1000
    min_reward_threshold: float = 0.7
    patience: int = 3
    
    def __post_init__(self):
        if self.difficulty_levels is None:
            self.difficulty_levels = ['easy', 'medium', 'hard', 'expert']

class CurriculumScheduler:
    """Manages curriculum learning progression."""
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.current_level_idx = 0
        self.steps_at_level = 0
        self.consecutive_good_rewards = 0
        
    def should_increase_difficulty(self, reward: float) -> bool:
        """Determine if difficulty should be increased."""
        self.steps_at_level += 1
        
        if reward >= self.config.min_reward_threshold:
            self.consecutive_good_rewards += 1
        else:
            self.consecutive_good_rewards = 0
            
        if (self.consecutive_good_rewards >= self.config.patience and
            self.steps_at_level >= self.config.steps_per_level):
            return True
        return False
        
    def increase_difficulty(self) -> Optional[str]:
        """Increase difficulty level if possible."""
        if self.current_level_idx < len(self.config.difficulty_levels) - 1:
            self.current_level_idx += 1
            self.steps_at_level = 0
            self.consecutive_good_rewards = 0
            return self.config.difficulty_levels[self.current_level_idx]
        return None
    
    @property
    def current_level(self) -> str:
        """Get current difficulty level."""
        return self.config.difficulty_levels[self.current_level_idx]

class AdvancedMathReasoningDataset(Dataset):
    """Enhanced dataset for training tree of thoughts on math reasoning tasks."""
    def __init__(self, 
                 data_path: str, 
                 tokenizer: Any, 
                 max_length: int = 512,
                 difficulty_level: str = 'medium'):
        with open(data_path, 'r') as f:
            all_data = json.load(f)
            
        # Filter by difficulty
        self.data = [
            item for item in all_data
            if item.get('metadata', {}).get('difficulty', 'medium') == difficulty_level
        ]
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.difficulty_level = difficulty_level
        
    def __len__(self):
        return len(self.data)
        
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize inputs
        input_encoded = self.tokenizer(
            item['question'],
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        # Process solution steps
        steps_encoded = []
        for step in item['solution_steps']:
            step_tokens = self.tokenizer(
                step,
                max_length=self.max_length // 4,  # Shorter for steps
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            steps_encoded.append(step_tokens)
        
        return {
            'input_ids': input_encoded['input_ids'].squeeze(0),
            'attention_mask': input_encoded['attention_mask'].squeeze(0),
            'solution': item['solution'],
            'steps': steps_encoded,
            'answer': item['answer'],
            'metadata': {
                'difficulty': item.get('difficulty', 'medium'),
                'category': item.get('category', 'general'),
                'num_steps': len(item['solution_steps'])
            }
        }

def collate_fn(batch: List[Dict]) -> Dict[str, torch.Tensor]:
    """Convert batch of examples to tensors."""
    return {
        'input_ids': torch.stack([item['input_text'] for item in batch]),
        'solutions': [item['solution'] for item in batch],
        'steps': [item['steps'] for item in batch],
        'answers': [item['answer'] for item in batch]
    }

class TrainingMonitor:
    """Monitor and log training progress."""
    
    def __init__(self, args: Any):
        self.args = args
        self.best_metrics = {'reward': float('-inf'), 'accuracy': 0.0}
        self.metrics_history = []
        
        if args.use_wandb:
            wandb.init(project="tree_of_thoughts_training", config=args)
    
    def log_metrics(self, metrics: Dict[str, float], step: int):
        """Log metrics to wandb and store in history."""
        self.metrics_history.append(metrics)
        
        # Update best metrics
        for metric in ['reward', 'accuracy']:
            if metrics.get(metric, float('-inf')) > self.best_metrics[metric]:
                self.best_metrics[metric] = metrics[metric]
                
        metrics['best_reward'] = self.best_metrics['reward']
        metrics['best_accuracy'] = self.best_metrics['accuracy']
        
        if self.args.use_wandb:
            wandb.log(metrics, step=step)
            
        # Log to console
        metrics_str = ' | '.join(f"{k}: {v:.4f}" for k, v in metrics.items())
        logger.info(f"Step {step}: {metrics_str}")
    
    def should_save_checkpoint(self, metrics: Dict[str, float]) -> bool:
        """Determine if current metrics warrant saving a checkpoint."""
        return metrics.get('reward', 0.0) >= self.best_metrics['reward']

def setup_training_args(args: Any) -> TrainingArguments:
    """Setup training arguments for the trainer."""
    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_steps=args.warmup_steps,
        eval_steps=args.eval_frequency,
        logging_steps=100,
        save_steps=args.save_frequency,
        fp16=True,
        logging_dir=str(Path(args.output_dir) / 'logs'),
        dataloader_num_workers=args.num_workers,
        remove_unused_columns=False
    )

    def update_difficulty(self, difficulty_level: str):
        """Update dataset to use problems of specified difficulty."""
        self.difficulty_level = difficulty_level
        with open(self.data_path, 'r') as f:
            all_data = json.load(f)
        self.data = [
            item for item in all_data
            if item.get('metadata', {}).get('difficulty', 'medium') == difficulty_level
        ]

def train(args: Any):
    # Initialize training monitor and curriculum scheduler
    monitor = TrainingMonitor(args)
    curriculum_config = CurriculumConfig(
        steps_per_level=args.curriculum_steps_per_level,
        min_reward_threshold=args.curriculum_reward_threshold,
        patience=args.curriculum_patience
    )
    curriculum = CurriculumScheduler(curriculum_config)
    
    # Load base model and tokenizer
    logger.info("Loading base model and tokenizer...")
    base_model = AutoModelForCausalLM.from_pretrained(args.model_name)
    tokenizer = base_model.tokenizer
    
    # Setup components
    logger.info("Setting up training components...")
    tree_config = TreeConfig(
        beam_width=args.beam_width,
        max_depth=args.max_depth,
        temperature=args.temperature,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        num_epochs=args.num_epochs,
        max_math_steps=args.max_math_steps,
        intermediate_steps=True,
        debug_mode=args.debug
    )
    
    training_args = setup_training_args(args)
    
    reward_config = RewardConfig(
        math_reasoning_weight=0.3,
        logical_coherence_weight=0.3,
        real_world_applicability_weight=0.2,
        solution_validity_weight=0.2,
        hidden_size=base_model.config.hidden_size
    )
    
    model = TreeOfThoughts(
        model=base_model,
        config=tree_config,
        reward_config=reward_config
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    # Initialize trainer with all components
    logger.info("Initializing trainer...")
    trainer = VishwamAIPretrainer(
        model_config=base_model.config.to_dict(),
        memory_module=NeuralMemory(
            memory_size=args.memory_size,
            hidden_dim=base_model.config.hidden_size
        ),
        tree_module=TreeOfThoughts(
            model=base_model,
            config=tree_config,
            reward_config=reward_config
        ),
        cache_module=CacheAugmentation(
            cache_size=args.cache_size,
            hidden_dim=base_model.config.hidden_size
        ),
        reward_config=reward_config,
        checkpoint_dir=args.output_dir,
        model=base_model,
        args=training_args,
        train_dataset=AdvancedMathReasoningDataset(
            args.train_data,
            tokenizer,
            max_length=args.max_length
        )
    )
    
    # Training loop with curriculum learning
    logger.info("Starting training loop with curriculum learning...")
    logger.info(f"Initial difficulty level: {curriculum.current_level}")
    global_step = 0
    train_iterator = tqdm(range(args.num_epochs), desc="Epoch")
    
    # Initialize dataset with starting difficulty
    dataset = AdvancedMathReasoningDataset(
        args.train_data,
        tokenizer,
        args.max_length,
        difficulty_level=curriculum.current_level
    )
    
    for epoch in train_iterator:
        epoch_metrics = defaultdict(float)
        num_batches = 0
        
        for batch_idx, batch in enumerate(trainer.get_train_dataloader()):
            batch = trainer._prepare_inputs(batch)
            
            with autocast():
                # Training step with all components
                outputs = trainer.training_step(trainer.model, batch)
                metrics = trainer._collect_training_stats(outputs, batch)
                
                # Accumulate metrics and check curriculum progression
                for k, v in metrics.items():
                    epoch_metrics[k] += v
                num_batches += 1
                
                reward = metrics.get('reward', 0.0)
                if curriculum.should_increase_difficulty(reward):
                    new_level = curriculum.increase_difficulty()
                    if new_level:
                        logger.info(f"\nIncreasing difficulty to: {new_level}")
                        dataset.update_difficulty(new_level)
                        # Update dataloader with new difficulty
                        trainer.train_dataset = dataset
                
                # Log batch metrics
                if global_step % args.log_frequency == 0:
                    batch_metrics = {
                        k: v / num_batches for k, v in epoch_metrics.items()
                    }
                    monitor.log_metrics(batch_metrics, global_step)
                
                # Save checkpoint if needed
                if (global_step % args.save_frequency == 0 and 
                    monitor.should_save_checkpoint(batch_metrics)):
                    trainer.save_model()
                    
                global_step += 1
        
            # End epoch processing
            if batch_idx == len(trainer.get_train_dataloader()) - 1:
                epoch_metrics = {
                    k: v / num_batches for k, v in epoch_metrics.items()
                }
                monitor.log_metrics({
                    **epoch_metrics,
                    'epoch': epoch
                }, global_step)
                
                # Evaluate on a few examples if in debug mode
                if args.debug:
                    _debug_evaluation(trainer, batch)
                
        # Save final epoch state
        trainer.save_model()

def _debug_evaluation(trainer: VishwamAIPretrainer, batch: Dict[str, torch.Tensor]):
    """Run debug evaluation on a batch."""
    trainer.model.eval()
    with torch.no_grad():
        outputs = trainer.model(**batch)
        nodes = trainer.tree_module._get_leaf_nodes(outputs.hidden_states[-1])
        for node in nodes[:3]:  # Show first 3 examples
            if node.text_output:
                logger.info(f"\nReasoning path:\n{node.text_output}")
    trainer.model.train()

def main():
    parser = argparse.ArgumentParser(description="Train Tree of Thoughts with advanced features")
    
    # Add curriculum learning parameters
    parser.add_argument('--curriculum_steps_per_level', type=int, default=1000,
                       help='Number of steps before considering difficulty increase')
    parser.add_argument('--curriculum_reward_threshold', type=float, default=0.7,
                       help='Minimum reward threshold for difficulty increase')
    parser.add_argument('--curriculum_patience', type=int, default=3,
                       help='Number of consecutive good rewards needed')
    # Model parameters
    parser.add_argument('--model_name', type=str, required=True, help='Base model name or path')
    parser.add_argument('--train_data', type=str, required=True, help='Path to training data')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    
    # Tree of Thoughts parameters
    parser.add_argument('--beam_width', type=int, default=4, help='Beam width for search')
    parser.add_argument('--max_depth', type=int, default=3, help='Maximum tree depth')
    parser.add_argument('--temperature', type=float, default=0.7, help='Sampling temperature')
    parser.add_argument('--max_math_steps', type=int, default=8, help='Maximum math reasoning steps')
    
    # Training parameters
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help='Gradient accumulation steps')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='Warmup steps')
    parser.add_argument('--max_length', type=int, default=512, help='Maximum sequence length')
    
    # Component parameters
    parser.add_argument('--memory_size', type=int, default=1024, help='Neural memory size')
    parser.add_argument('--cache_size', type=int, default=512, help='Cache size')
    
    # System parameters
    parser.add_argument('--num_workers', type=int, default=4, help='Number of dataloader workers')
    parser.add_argument('--log_frequency', type=int, default=100, help='Logging frequency')
    parser.add_argument('--save_frequency', type=int, default=1000, help='Model saving frequency')
    parser.add_argument('--eval_frequency', type=int, default=500, help='Evaluation frequency')
    
    # Monitoring
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
