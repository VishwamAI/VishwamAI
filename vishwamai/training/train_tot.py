"""
Tree of Thoughts (ToT) training script for VishwamAI.
Extends CoT training with tree search and thought evaluation.
"""

import os
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import aim
import logging
from tqdm import tqdm
from typing import Dict, Optional, Any, Tuple, List
import numpy as np

from vishwamai.models.tot_model import ToTModel, ThoughtNode
from vishwamai.training.dataset_loader import VishwamAIDataset, create_dataloader
from vishwamai.training.train_cot import CoTTrainer
from vishwamai.optimisation.memory_optimization import MemoryOptimizer

logger = logging.getLogger(__name__)

class ToTTrainer(CoTTrainer):
    """
    Specialized trainer for Tree of Thoughts models.
    Extends CoT trainer with tree search and thought evaluation.
    """
    
    def __init__(
        self,
        model: ToTModel,
        train_dataset: VishwamAIDataset,
        val_dataset: Optional[VishwamAIDataset] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Initialize ToT trainer"""
        # Update default config with ToT-specific settings
        default_config = {
            'max_tree_depth': 5,        # Maximum depth of thought tree
            'beam_size': 5,             # Beam size for tree search
            'max_branches': 3,          # Maximum branches per node
            'min_prob_threshold': 0.1,  # Minimum probability for valid thoughts
            'search_strategy': 'bfs',   # 'bfs' or 'dfs'
            'tree_loss_weight': 0.3,    # Weight for tree structure loss
            'experiment_name': 'vishwamai_tot_training'  # Aim experiment name
        }
        if config:
            default_config.update(config)
            
        super().__init__(model, train_dataset, val_dataset, default_config)
        
        # Add tree-specific metrics
        self.thought_metrics.update({
            'tree_loss': 0.0,
            'avg_tree_depth': 0.0,
            'avg_branching_factor': 0.0,
            'successful_paths': 0.0
        })
        
    def setup_aim_logging(self):
        """Initialize Aim experiment tracking with ToT-specific settings"""
        self.aim_run = aim.Run(
            experiment=self.config['experiment_name'],
            log_system_params=True
        )
        
        # Log configuration including ToT-specific parameters
        self.aim_run["hparams"] = self.config
        
        # Set descriptive run name
        self.aim_run.name = f"tot_{self.config['experiment_name']}_{aim.Run.generate_run_hash()}"
        
        # Create metric contexts for different types of metrics
        self.aim_run.create_context("thought_metrics")
        self.aim_run.create_context("tree_metrics")
        self.aim_run.create_context("answer_metrics")
        
        # Create context for tree visualizations
        self.aim_run.create_context("tree_visualizations")
        
    def _log_tree_visualization(self, thought_tree: Dict, step: int):
        """Log tree structure visualization to Aim"""
        if not self.is_main_process:
            return
            
        # Convert tree to a format suitable for visualization
        def process_tree_for_viz(node, depth=0):
            if isinstance(node, str):
                return {
                    "name": node[:50] + "..." if len(node) > 50 else node,
                    "depth": depth
                }
            
            children = []
            for key, value in node.items():
                child = process_tree_for_viz(value, depth + 1)
                if child:
                    children.append(child)
                    
            return {
                "name": f"Depth {depth}",
                "children": children,
                "depth": depth
            }
            
        tree_viz = process_tree_for_viz(thought_tree)
        
        # Log tree visualization with Aim
        self.aim_run.track(
            aim.Figure(tree_viz),  # Aim will automatically convert to tree visualization
            name="thought_tree_structure",
            step=step,
            context="tree_visualizations"
        )
        
    def compute_tree_loss(
        self,
        thought_tree: Dict,
        logits: torch.Tensor,
        target_tree: Optional[Dict] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute loss for tree structure and thought coherence.
        
        Args:
            thought_tree: Generated thought tree
            logits: Model output logits
            target_tree: Target thought tree if available
            
        Returns:
            loss: Tree structure loss
            metrics: Tree-specific metrics
        """
        # Initialize metrics
        metrics = {
            'tree_depth': 0,
            'branching_factor': 0,
            'path_success': 0
        }
        
        # Compute tree statistics
        def process_tree(node, depth=0):
            if isinstance(node, str):
                return depth, 0, 1
            
            max_depth = depth
            total_branches = 0
            num_nodes = 1
            
            for child in node.values():
                child_depth, child_branches, child_nodes = process_tree(child, depth + 1)
                max_depth = max(max_depth, child_depth)
                total_branches += child_branches + 1
                num_nodes += child_nodes
                
            return max_depth, total_branches, num_nodes
            
        tree_depth, total_branches, num_nodes = process_tree(thought_tree)
        
        # Update metrics
        metrics['tree_depth'] = tree_depth
        metrics['branching_factor'] = total_branches / max(1, num_nodes - 1)
        
        # Compute tree structure loss
        if target_tree is not None:
            # Compare with target tree structure
            structure_loss = self._compute_tree_similarity(thought_tree, target_tree)
        else:
            # Encourage balanced tree growth
            ideal_branching = self.config['max_branches'] / 2
            branching_loss = torch.abs(torch.tensor(metrics['branching_factor'] - ideal_branching))
            depth_loss = torch.abs(torch.tensor(tree_depth - self.config['max_tree_depth']))
            structure_loss = (branching_loss + depth_loss) / 2
            
        return structure_loss, metrics
        
    def train_step(self, batch: Dict[str, torch.Tensor]) -> float:
        """Execute single ToT training step"""
        self.model.train()
        
        # Move batch to device
        input_ids = batch['input_ids'].to(self.device)
        target_ids = batch['target_ids'].to(self.device)
        
        # Get thought tree if available
        thought_tree = batch['metadata'].get('thought_tree', None)
        
        # Forward pass with mixed precision
        with autocast(enabled=self.config['fp16']):
            # Standard CoT loss
            logits = self.model(input_ids)
            cot_loss, cot_metrics = self.compute_loss(logits, target_ids)
            
            # Tree search and loss
            if thought_tree is not None:
                generated_tree = self.model.solve_with_tot(
                    input_ids,
                    search_method=self.config['search_strategy'],
                    b=self.config['beam_size']
                )
                tree_loss, tree_metrics = self.compute_tree_loss(
                    generated_tree,
                    logits,
                    thought_tree
                )
                
                # Log tree visualization periodically
                if self.step % self.config['logging_steps'] == 0:
                    self._log_tree_visualization(generated_tree, self.step)
                
                # Combine losses
                loss = (
                    (1 - self.config['tree_loss_weight']) * cot_loss +
                    self.config['tree_loss_weight'] * tree_loss
                )
                
                # Update metrics
                self.thought_metrics.update({
                    'tree_loss': tree_loss.item(),
                    'avg_tree_depth': tree_metrics['tree_depth'],
                    'avg_branching_factor': tree_metrics['branching_factor']
                })
                
                # Log tree-specific metrics with Aim
                if self.is_main_process:
                    for k, v in tree_metrics.items():
                        self.aim_run.track(
                            v,
                            name=f'train/tree_{k}',
                            step=self.step,
                            epoch=self.epoch,
                            context="tree_metrics"
                        )
            else:
                loss = cot_loss
                
            loss = loss / self.config['grad_acc_steps']
            
        # Update thought metrics
        self.thought_metrics.update(cot_metrics)
        
        # Log metrics with Aim under appropriate contexts
        if self.is_main_process and self.step % self.config['logging_steps'] == 0:
            for k, v in self.thought_metrics.items():
                context = (
                    "tree_metrics" if "tree" in k
                    else "thought_metrics" if "thought" in k
                    else "answer_metrics"
                )
                self.aim_run.track(
                    v,
                    name=f'train/{k}',
                    step=self.step,
                    epoch=self.epoch,
                    context=context
                )
        
        # Backward pass with gradient scaling
        if self.config['fp16']:
            self.scaler.scale(loss).backward()
            if (self.step + 1) % self.config['grad_acc_steps'] == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
        else:
            loss.backward()
            if (self.step + 1) % self.config['grad_acc_steps'] == 0:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.optimizer.zero_grad()
                
        return loss.item() * self.config['grad_acc_steps']
        
    @torch.no_grad()
    def evaluate(self) -> float:
        """Run evaluation with tree-specific metrics"""
        if not self.val_dataset:
            return 0.0
            
        self.model.eval()
        total_loss = 0
        total_metrics = {k: 0.0 for k in self.thought_metrics.keys()}
        total_steps = 0
        
        for batch in self.val_dataloader:
            input_ids = batch['input_ids'].to(self.device)
            target_ids = batch['target_ids'].to(self.device)
            thought_tree = batch['metadata'].get('thought_tree', None)
            
            # Forward pass
            logits = self.model(input_ids)
            cot_loss, cot_metrics = self.compute_loss(logits, target_ids)
            
            # Tree evaluation if tree data available
            if thought_tree is not None:
                generated_tree = self.model.solve_with_tot(
                    input_ids,
                    search_method=self.config['search_strategy'],
                    b=self.config['beam_size']
                )
                tree_loss, tree_metrics = self.compute_tree_loss(
                    generated_tree,
                    logits,
                    thought_tree
                )
                
                # Combined loss
                loss = (
                    (1 - self.config['tree_loss_weight']) * cot_loss +
                    self.config['tree_loss_weight'] * tree_loss
                )
                
                # Update metrics
                total_metrics.update({
                    'tree_loss': tree_loss.item(),
                    'avg_tree_depth': tree_metrics['tree_depth'],
                    'avg_branching_factor': tree_metrics['branching_factor']
                })
            else:
                loss = cot_loss
                
            total_loss += loss.item()
            total_metrics.update(cot_metrics)
            total_steps += 1
            
        # Compute averages
        avg_loss = total_loss / total_steps
        avg_metrics = {k: v / total_steps for k, v in total_metrics.items()}
        
        # Log validation metrics with Aim
        if self.is_main_process:
            self.aim_run.track(
                avg_loss,
                name='val/loss',
                step=self.step,
                epoch=self.epoch
            )
            
            # Log detailed validation metrics under appropriate contexts
            for k, v in avg_metrics.items():
                context = (
                    "tree_metrics" if "tree" in k
                    else "thought_metrics" if "thought" in k
                    else "answer_metrics"
                )
                self.aim_run.track(
                    v,
                    name=f'val/{k}',
                    step=self.step,
                    epoch=self.epoch,
                    context=context
                )
            
        return avg_loss

def main():
    """Main training function"""
    # Load config
    config = {
        'batch_size': 8,      # Even smaller batch size for ToT
        'grad_acc_steps': 8,  # More gradient accumulation steps
        'learning_rate': 2e-5,
        'warmup_steps': 2000,
        'max_steps': 100000,
        'save_steps': 1000,
        'eval_steps': 500,
        'local_rank': int(os.environ.get('LOCAL_RANK', -1)),
        'fp16': True,
        'checkpoint_dir': 'checkpoints',
        'max_tree_depth': 5,
        'beam_size': 5,
        'max_branches': 3,
        'search_strategy': 'bfs',
        'tree_loss_weight': 0.3,
        'experiment_name': 'vishwamai_tot_training'
    }
    
    # Initialize model
    model = ToTModel(
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        ff_dim=3072,
        vocab_size=50000
    )
    
    # Load datasets
    train_dataset = VishwamAIDataset(
        data_path='path/to/train.json',
        tokenizer=None,  # Add your tokenizer here
        mode='tot'
    )
    
    val_dataset = VishwamAIDataset(
        data_path='path/to/val.json',
        tokenizer=None,  # Add your tokenizer here
        mode='tot'
    )
    
    # Initialize trainer
    trainer = ToTTrainer(model, train_dataset, val_dataset, config)
    
    # Start training
    trainer.train()

if __name__ == "__main__":
    main()