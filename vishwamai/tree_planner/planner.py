"""
Tree-based planner implementation
"""
from typing import List, Dict, Any, Optional, Tuple
import torch
import torch.nn as nn

from .node import Node
from .search import TreeSearch

class TreePlanner(nn.Module):
    """
    Tree-based planner that performs hierarchical planning using learned models
    """
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_tree_depth: int = 5,
        num_samples: int = 10,
        temperature: float = 1.0
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_tree_depth = max_tree_depth
        self.num_samples = num_samples
        
        # Tree search
        self.search = TreeSearch(
            temperature=temperature,
            num_simulations=num_samples
        )
        
        # Planning networks
        self.state_encoder = nn.Linear(hidden_size, hidden_size)
        self.action_scorer = nn.Linear(hidden_size, hidden_size)
        self.value_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
        
        # Action generation
        self.action_generator = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        
        self.dropout = nn.Dropout(dropout)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        """
        Encode state into latent representation
        
        Args:
            state: State tensor
            
        Returns:
            Encoded state representation
        """
        return self.dropout(self.state_encoder(state))

    def predict_value(self, state: torch.Tensor) -> torch.Tensor:
        """
        Predict value for a state
        
        Args:
            state: Encoded state tensor
            
        Returns:
            Value prediction
        """
        return self.value_net(state).squeeze(-1)

    def generate_actions(
        self,
        state: torch.Tensor,
        num_actions: int
    ) -> Tuple[torch.Tensor, List[str]]:
        """
        Generate possible actions from current state
        
        Args:
            state: Encoded state tensor
            num_actions: Number of actions to generate
            
        Returns:
            Tuple of (action_states, action_descriptions)
        """
        batch_size = state.size(0)
        
        # Generate multiple action proposals
        action_hidden = self.action_generator(state)
        action_hidden = action_hidden.unsqueeze(1).expand(-1, num_actions, -1)
        
        # Add random noise for diversity
        noise = torch.randn_like(action_hidden) * 0.1
        action_hidden = action_hidden + noise
        
        # Score actions
        action_scores = self.action_scorer(action_hidden)
        action_states = action_hidden + action_scores
        
        # Generate text descriptions (placeholder)
        action_descs = [f"Action_{i}" for i in range(num_actions)]
        
        return action_states, action_descs

    def plan(
        self,
        initial_state: torch.Tensor,
        max_steps: Optional[int] = None
    ) -> Tuple[List[str], torch.Tensor]:
        """
        Generate plan from initial state
        
        Args:
            initial_state: Initial state tensor
            max_steps: Maximum number of planning steps
            
        Returns:
            Tuple of (action_sequence, final_state)
        """
        if max_steps is None:
            max_steps = self.max_tree_depth
            
        # Encode initial state
        state = self.encode_state(initial_state)
        
        # Create root node
        root = Node(hidden_state=state)
        
        def model_fn(node_state, num_children):
            """Generate child states/actions"""
            states, actions = self.generate_actions(
                node_state.unsqueeze(0),
                num_children
            )
            return states.squeeze(0), actions
            
        def value_fn(node_state):
            """Evaluate node value"""
            return self.predict_value(node_state.unsqueeze(0))
        
        # Perform tree search
        best_children, action_probs = self.search.search(
            root=root,
            model_fn=model_fn,
            value_fn=value_fn
        )
        
        # Extract best action sequence
        if not best_children:
            return [], initial_state
            
        best_child = max(
            best_children,
            key=lambda n: n.score
        )
        action_sequence = best_child.get_path()
        final_state = best_child.hidden_state
        
        return action_sequence, final_state

    def forward(
        self,
        states: torch.Tensor,
        num_steps: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Generate plans for batch of states
        
        Args:
            states: Batch of state tensors
            num_steps: Maximum number of planning steps
            
        Returns:
            Dictionary with plans and metrics
        """
        batch_size = states.size(0)
        
        # Generate plans for each state
        all_actions = []
        all_states = []
        
        for i in range(batch_size):
            actions, final_state = self.plan(
                states[i],
                max_steps=num_steps
            )
            all_actions.append(actions)
            all_states.append(final_state)
            
        # Stack final states
        final_states = torch.stack(all_states)
            
        return {
            'actions': all_actions,
            'final_states': final_states
        }
