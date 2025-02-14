import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
from transformers import PreTrainedModel
import numpy as np

@dataclass
class TreeConfig:
    """Configuration for Tree of Thoughts implementation."""
    beam_width: int = 4
    max_depth: int = 3
    temperature: float = 0.7
    top_k: int = 50
    pruning_threshold: float = 0.1
    rewrite_factor: float = 0.3
    hidden_size: int = 8192

class TreeNode:
    """Node in the Tree of Thoughts structure."""
    def __init__(self, state: torch.Tensor, score: float = 0.0):
        self.state = state
        self.score = score
        self.children: List[TreeNode] = []
        self.parent: Optional[TreeNode] = None
        self.depth: int = 0
        
    def add_child(self, child: 'TreeNode'):
        """Add a child node and update its depth."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

class TreeOfThoughts(nn.Module):
    """Tree of Thoughts implementation for structured reasoning."""
    
    def __init__(self, 
                 model: PreTrainedModel,
                 config: Optional[TreeConfig] = None):
        super().__init__()
        self.config = config or TreeConfig()
        self.base_model = model
        
        # Reasoning networks
        self.state_evaluator = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, 1)
        )
        
        self.state_expander = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * self.config.beam_width)
        )
        
        self.state_refiner = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
        )
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Process hidden states through tree-structured reasoning."""
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        
        # Initialize root nodes for each batch item
        roots = [TreeNode(state) for state in hidden_states]
        
        # Build trees through iterative reasoning
        for depth in range(self.config.max_depth):
            level_nodes = [node for node in self._get_level_nodes(roots, depth)]
            
            if not level_nodes:
                break
                
            # Expand nodes
            expanded_states = []
            for node in level_nodes:
                expanded = self.state_expander(node.state)
                expanded = expanded.view(-1, self.config.beam_width, self.config.hidden_size)
                expanded_states.append(expanded)
            
            expanded_states = torch.cat(expanded_states, dim=0)
            
            # Evaluate expanded states
            scores = self.state_evaluator(expanded_states.view(-1, self.config.hidden_size))
            scores = scores.view(-1, self.config.beam_width)
            
            # Select best children
            topk_scores, topk_indices = torch.topk(
                scores, k=min(self.config.beam_width, scores.size(1)), dim=1
            )
            
            # Create child nodes with pruning
            for i, node in enumerate(level_nodes):
                node_scores = topk_scores[i]
                node_states = expanded_states[i][topk_indices[i]]
                
                for score, state in zip(node_scores, node_states):
                    if score > self.config.pruning_threshold:
                        child = TreeNode(state, score.item())
                        node.add_child(child)
        
        # Extract best paths and refine final states
        best_paths = self._extract_best_paths(roots)
        refined_states = self._refine_states(best_paths, hidden_states)
        
        return refined_states
    
    def _get_level_nodes(self, roots: List[TreeNode], depth: int) -> List[TreeNode]:
        """Get all nodes at a specific depth."""
        if depth == 0:
            return roots
            
        nodes = []
        def collect_nodes(node: TreeNode, current_depth: int):
            if current_depth == depth:
                nodes.append(node)
            else:
                for child in node.children:
                    collect_nodes(child, current_depth + 1)
                    
        for root in roots:
            collect_nodes(root, 0)
            
        return nodes
    
    def _extract_best_paths(self, roots: List[TreeNode]) -> List[List[torch.Tensor]]:
        """Extract best reasoning paths from trees."""
        paths = []
        
        def get_best_path(node: TreeNode) -> List[torch.Tensor]:
            path = [node.state]
            if node.children:
                best_child = max(node.children, key=lambda x: x.score)
                path.extend(get_best_path(best_child))
            return path
        
        for root in roots:
            paths.append(get_best_path(root))
            
        return paths
    
    def _refine_states(self, paths: List[List[torch.Tensor]], 
                      original_states: torch.Tensor) -> torch.Tensor:
        """Refine states using reasoning paths."""
        refined_states = []
        
        for path, orig_state in zip(paths, original_states):
            # Combine all states in the path
            path_tensor = torch.stack(path)
            
            # Get final refined state
            path_encoding = path_tensor.mean(dim=0)
            refined = self.state_refiner(
                torch.cat([orig_state, path_encoding], dim=-1)
            )
            
            # Interpolate with original state
            refined = (1 - self.config.rewrite_factor) * orig_state + \
                     self.config.rewrite_factor * refined
            
            refined_states.append(refined)
            
        return torch.stack(refined_states)
    
    def save_pretrained(self, save_path: str):
        """Save tree of thoughts components."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, f"{save_path}/tree_of_thoughts.pt")
        
    @classmethod
    def from_pretrained(cls, load_path: str, model: PreTrainedModel):
        """Load tree of thoughts components."""
        checkpoint = torch.load(f"{load_path}/tree_of_thoughts.pt")
        instance = cls(model=model, config=checkpoint['config'])
        instance.load_state_dict(checkpoint['state_dict'])
        return instance
