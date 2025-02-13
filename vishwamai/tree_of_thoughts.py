import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple, Dict
from dataclasses import dataclass
import math

@dataclass
class TreeConfig:
    max_branches: int = 4  # Maximum number of branches to explore per node
    max_depth: int = 3     # Maximum depth of the reasoning tree
    beam_width: int = 2    # Number of best paths to maintain
    temperature: float = 0.8  # Temperature for sampling
    min_score_diff: float = 0.1  # Minimum score difference to consider a path different

class ThoughtNode:
    """Represents a node in the reasoning tree"""
    def __init__(self, 
                 hidden_state: torch.Tensor,
                 score: float,
                 parent: Optional['ThoughtNode'] = None,
                 depth: int = 0):
        self.hidden_state = hidden_state
        self.score = score
        self.parent = parent
        self.depth = depth
        self.children: List['ThoughtNode'] = []
        
    def add_child(self, hidden_state: torch.Tensor, score: float) -> 'ThoughtNode':
        child = ThoughtNode(hidden_state, score, self, self.depth + 1)
        self.children.append(child)
        return child
        
    def get_path_to_root(self) -> List[torch.Tensor]:
        """Get the sequence of hidden states from root to this node"""
        path = []
        current = self
        while current is not None:
            path.append(current.hidden_state)
            current = current.parent
        return list(reversed(path))

class TreeOfThoughts(nn.Module):
    """
    Implements tree-based reasoning by exploring multiple thought paths
    and selecting the most promising ones.
    """
    def __init__(self, 
                 hidden_size: int,
                 config: TreeConfig):
        super().__init__()
        self.hidden_size = hidden_size
        self.config = config
        
        # Evaluation network to score reasoning paths
        self.evaluator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid()
        )
        
        # Thought generation networks
        self.thought_proj = nn.Linear(hidden_size, hidden_size)
        self.thought_combine = nn.Linear(hidden_size * 2, hidden_size)
        
        # Value estimation for path selection
        self.value_estimator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1),
            nn.Tanh()
        )
        
    def evaluate_state(self, state: torch.Tensor) -> torch.Tensor:
        """Evaluate the promise of a particular reasoning state"""
        return self.evaluator(state)
    
    def generate_thoughts(self, 
                         current_state: torch.Tensor,
                         num_branches: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate multiple possible next thoughts from current state"""
        # Project current state
        projected = self.thought_proj(current_state)
        
        # Generate multiple thought vectors through different linear projections
        thoughts = []
        for _ in range(num_branches):
            # Create a unique projection for this branch
            branch_proj = nn.Linear(self.hidden_size, self.hidden_size, device=current_state.device)
            thought = F.gelu(branch_proj(projected))
            thoughts.append(thought)
        
        thoughts = torch.stack(thoughts, dim=1)  # [batch_size, num_branches, hidden_size]
        
        # Generate scores for each thought
        scores = self.evaluate_state(thoughts.view(-1, self.hidden_size))
        scores = scores.view(-1, num_branches)
        
        return thoughts, scores
    
    def expand_node(self, 
                   node: ThoughtNode, 
                   temperature: float = 1.0) -> List[ThoughtNode]:
        """Expand a node by generating multiple possible next thoughts"""
        thoughts, scores = self.generate_thoughts(
            node.hidden_state,
            self.config.max_branches
        )
        
        # Apply temperature to scores
        scores = scores / temperature
        
        # Create child nodes
        children = []
        for i in range(thoughts.size(1)):
            thought = thoughts[:, i]
            score = scores[:, i].mean().item()
            
            # Only add child if score is significantly different
            if not children or min(abs(score - c.score) for c in children) >= self.config.min_score_diff:
                child = node.add_child(thought, score)
                children.append(child)
                
                # Limit number of children
                if len(children) >= self.config.beam_width:
                    break
                
        return children
    
    def search_best_path(self, 
                        initial_state: torch.Tensor,
                        attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Perform tree search to find the most promising reasoning path.
        Uses beam search to maintain multiple promising paths.
        """
        # Initialize root node
        root_score = self.evaluate_state(initial_state).mean().item()
        root = ThoughtNode(initial_state, root_score)
        
        # Initialize beam with root node
        beam = [root]
        
        # Expand tree up to max depth
        for depth in range(self.config.max_depth):
            candidates = []
            
            # Expand each node in current beam
            for node in beam:
                children = self.expand_node(
                    node,
                    temperature=self.config.temperature
                )
                candidates.extend(children)
            
            if not candidates:
                break
                
            # Sort candidates by score and select top-k for beam
            candidates.sort(key=lambda x: x.score, reverse=True)
            beam = candidates[:self.config.beam_width]
        
        # Select best final node
        best_node = max(beam, key=lambda x: x.score)
        
        # Return sequence of hidden states along best path
        path_states = best_node.get_path_to_root()
        return torch.stack(path_states, dim=1)  # [batch_size, path_length, hidden_size]
    
    def forward(self,
                hidden_states: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Process hidden states through tree of thoughts reasoning.
        
        Args:
            hidden_states: Input tensor of shape [batch_size, seq_length, hidden_size]
            attention_mask: Optional attention mask
            
        Returns:
            Processed hidden states incorporating best reasoning path
        """
        batch_size, seq_len, _ = hidden_states.shape
        
        # Process each sequence position through tree search
        outputs = []
        
        for pos in range(seq_len):
            current_state = hidden_states[:, pos]
            
            # Perform tree search from current state
            path_states = self.search_best_path(current_state, attention_mask)
            
            # Use final state in best path
            outputs.append(path_states[:, -1])
            
        # Combine outputs
        output = torch.stack(outputs, dim=1)
        
        # Residual connection
        return output + hidden_states
