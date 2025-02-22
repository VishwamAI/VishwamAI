"""
Tree node implementation for hierarchical planning
"""
from typing import List, Optional, Dict, Any
import torch

class Node:
    """
    Tree node representing a planning state or action
    """
    def __init__(
        self,
        hidden_state: torch.Tensor,
        parent: Optional['Node'] = None,
        action: Optional[str] = None,
        score: float = 0.0
    ):
        self.hidden_state = hidden_state  # Encoded state representation
        self.parent = parent  # Parent node
        self.children: List[Node] = []  # Child nodes
        self.action = action  # Action that led to this node
        self.score = score  # Node evaluation score
        self.visit_count = 0  # Number of times node was visited during search
        
        # Add to parent's children if parent exists
        if parent is not None:
            parent.add_child(self)
            
    def add_child(self, child: 'Node') -> None:
        """Add a child node"""
        self.children.append(child)
        child.parent = self
        
    def remove_child(self, child: 'Node') -> None:
        """Remove a child node"""
        self.children.remove(child)
        child.parent = None
        
    def is_leaf(self) -> bool:
        """Check if node is a leaf node"""
        return len(self.children) == 0
        
    def is_root(self) -> bool:
        """Check if node is the root node"""
        return self.parent is None
        
    def depth(self) -> int:
        """Get depth of node in tree"""
        if self.is_root():
            return 0
        return self.parent.depth() + 1
    
    def get_path(self) -> List[str]:
        """Get sequence of actions from root to this node"""
        if self.is_root():
            return []
        return self.parent.get_path() + [self.action]
    
    def get_siblings(self) -> List['Node']:
        """Get sibling nodes"""
        if self.is_root():
            return []
        return [child for child in self.parent.children if child != self]
    
    def update_score(self, new_score: float) -> None:
        """Update node score and visit count"""
        # Incremental average
        self.visit_count += 1
        self.score = ((self.score * (self.visit_count - 1)) + new_score) / self.visit_count
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert node to dictionary representation"""
        return {
            'action': self.action,
            'score': self.score,
            'visit_count': self.visit_count,
            'children': [child.to_dict() for child in self.children]
        }
        
    def __repr__(self) -> str:
        return f"Node(action={self.action}, score={self.score:.3f}, visits={self.visit_count})"
