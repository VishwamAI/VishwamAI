"""
Tree search algorithms for planning
"""
from typing import List, Optional, Tuple
import torch
import numpy as np
from .node import Node

class TreeSearch:
    """
    Tree search implementation with MCTS-like exploration
    """
    def __init__(
        self,
        c_puct: float = 1.0,
        num_simulations: int = 100,
        temperature: float = 1.0
    ):
        self.c_puct = c_puct  # Exploration constant
        self.num_simulations = num_simulations  # Number of search simulations
        self.temperature = temperature  # Temperature for action selection
        
    def _ucb_score(self, parent: Node, child: Node) -> float:
        """
        Calculate Upper Confidence Bound (UCB) score for node selection
        
        Args:
            parent: Parent node
            child: Child node to evaluate
            
        Returns:
            UCB score combining exploitation and exploration terms
        """
        # Exploitation term
        q_value = child.score
        
        # Exploration term
        total_visits = sum(n.visit_count for n in parent.children)
        exploration = (self.c_puct * np.sqrt(total_visits) / 
                      (1 + child.visit_count))
                      
        return q_value + exploration
        
    def select_node(self, node: Node) -> Node:
        """
        Select most promising node to explore using UCB
        
        Args:
            node: Current node
            
        Returns:
            Selected node to explore next
        """
        while not node.is_leaf():
            # Select child with highest UCB score
            node = max(
                node.children,
                key=lambda n: self._ucb_score(node, n)
            )
        return node
        
    def expand_node(
        self,
        node: Node,
        model_fn,
        num_children: int = 5
    ) -> List[Node]:
        """
        Expand node by generating children
        
        Args:
            node: Node to expand
            model_fn: Function to generate child states/actions
            num_children: Number of children to generate
            
        Returns:
            List of generated child nodes
        """
        # Get child states/actions from model
        child_states, child_actions = model_fn(node.hidden_state, num_children)
        
        # Create child nodes
        children = []
        for state, action in zip(child_states, child_actions):
            child = Node(
                hidden_state=state,
                parent=node,
                action=action
            )
            children.append(child)
            
        return children
        
    def evaluate_node(
        self,
        node: Node,
        value_fn
    ) -> float:
        """
        Evaluate node value using value network
        
        Args:
            node: Node to evaluate
            value_fn: Value network function
            
        Returns:
            Estimated value of node
        """
        value = value_fn(node.hidden_state)
        node.update_score(value.item())
        return value.item()
        
    def backpropagate(
        self,
        node: Node,
        value: float
    ) -> None:
        """
        Backpropagate value through ancestors
        
        Args:
            node: Leaf node
            value: Value to backpropagate
        """
        while node is not None:
            node.update_score(value)
            node = node.parent
            
    def search(
        self,
        root: Node,
        model_fn,
        value_fn
    ) -> Tuple[List[Node], List[float]]:
        """
        Perform tree search to find best action sequence
        
        Args:
            root: Root node
            model_fn: Function to generate child states/actions
            value_fn: Value network function
            
        Returns:
            Tuple of (best_children, action_probs)
        """
        for _ in range(self.num_simulations):
            # Selection
            leaf = self.select_node(root)
            
            # Expansion
            if not leaf.is_leaf():
                continue
            children = self.expand_node(leaf, model_fn)
            
            if not children:
                continue
                
            # Evaluation
            value = self.evaluate_node(children[0], value_fn)
            
            # Backpropagation
            self.backpropagate(children[0], value)
            
        # Calculate action probabilities
        visit_counts = np.array([
            child.visit_count for child in root.children
        ])
        action_probs = visit_counts ** (1.0 / self.temperature)
        action_probs = action_probs / action_probs.sum()
        
        return root.children, action_probs.tolist()
