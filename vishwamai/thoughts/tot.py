"""Tree of Thoughts implementation for VishwamAI."""

from typing import List, Optional, Tuple, Dict, Any
import jax
import jax.numpy as jnp
import numpy as np
from dataclasses import dataclass
from vishwamai.model import VishwamAI

@dataclass
class ThoughtNode:
    """Represents a node in the tree of thoughts."""
    thought: str
    value: float
    children: List['ThoughtNode']
    parent: Optional['ThoughtNode']
    depth: int

class TreeOfThoughts:
    """Implements tree-based reasoning for complex problem solving."""
    
    def __init__(
        self,
        model: VishwamAI,
        max_branches: int = 3,
        max_depth: int = 3,
        beam_width: int = 5,
        temperature: float = 0.7
    ):
        """Initialize Tree of Thoughts.
        
        Args:
            model: Base language model
            max_branches: Maximum branching factor at each node
            max_depth: Maximum tree depth
            beam_width: Number of paths to maintain
            temperature: Sampling temperature
        """
        self.model = model
        self.max_branches = max_branches
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.temperature = temperature
        
    def generate_thoughts(
        self,
        context: str,
        num_thoughts: int = 3,
        temperature: Optional[float] = None
    ) -> List[str]:
        """Generate multiple potential thought branches.
        
        Args:
            context: Current context/prompt
            num_thoughts: Number of thoughts to generate
            temperature: Optional override for sampling temperature
            
        Returns:
            List of generated thoughts
        """
        temperature = temperature or self.temperature
        
        # Prepare input
        inputs = self.model.tokenizer.encode(
            context,
            return_tensors="jax",
            max_length=self.model.config.max_seq_len,
            truncation=True
        )
        
        thoughts = []
        for _ in range(num_thoughts):
            outputs = self.model.generate(
                input_ids=inputs,
                max_length=self.model.config.max_seq_len,
                temperature=temperature,
                top_p=0.9,
                do_sample=True,
                num_return_sequences=1
            )
            
            # Decode and clean up output
            thought = self.model.tokenizer.decode(
                outputs[0],
                skip_special_tokens=True
            )
            thoughts.append(thought.strip())
            
        return thoughts
    
    def estimate_value(
        self,
        thought_sequence: List[str],
        objective: str
    ) -> float:
        """Estimate the value/quality of a thought sequence.
        
        Args:
            thought_sequence: Sequence of thoughts to evaluate
            objective: Goal/objective for evaluation
            
        Returns:
            Estimated value between 0 and 1
        """
        # Combine thoughts and objective for evaluation
        eval_prompt = (
            f"Objective: {objective}\n\n"
            f"Thought sequence:\n"
            + "\n".join(f"{i+1}. {t}" for i, t in enumerate(thought_sequence))
            + "\n\nOn a scale of 0 to 1, rate how well this thought sequence achieves the objective."
        )
        
        inputs = self.model.tokenizer.encode(
            eval_prompt,
            return_tensors="jax",
            max_length=self.model.config.max_seq_len,
            truncation=True
        )
        
        outputs = self.model.generate(
            input_ids=inputs,
            max_length=20,  # Short output for rating
            temperature=0.1  # Low temperature for more consistent evaluation
        )
        
        # Extract numerical rating
        response = self.model.tokenizer.decode(outputs[0], skip_special_tokens=True)
        try:
            # Find first number in response
            value = float(next(
                float(s) for s in response.split() 
                if s.replace(".", "").isdigit()
            ))
            # Ensure value is between 0 and 1
            return max(0.0, min(1.0, value))
        except (StopIteration, ValueError):
            return 0.0  # Default to 0 if no valid rating found
    
    def expand_node(
        self,
        node: ThoughtNode,
        context: str,
        objective: str
    ) -> List[ThoughtNode]:
        """Expand a node by generating child thoughts.
        
        Args:
            node: Node to expand
            context: Current context
            objective: Goal/objective
            
        Returns:
            List of child nodes
        """
        if node.depth >= self.max_depth:
            return []
            
        # Generate new thoughts considering parent thought
        expanded_context = (
            f"{context}\n"
            f"Current thought: {node.thought}\n"
            f"Given this, generate the next step:"
        )
        
        new_thoughts = self.generate_thoughts(
            expanded_context,
            num_thoughts=self.max_branches
        )
        
        # Create child nodes
        children = []
        for thought in new_thoughts:
            thought_sequence = self._get_thought_sequence(node) + [thought]
            value = self.estimate_value(thought_sequence, objective)
            
            child = ThoughtNode(
                thought=thought,
                value=value,
                children=[],
                parent=node,
                depth=node.depth + 1
            )
            children.append(child)
            
        node.children = children
        return children
    
    def _get_thought_sequence(self, node: ThoughtNode) -> List[str]:
        """Get the sequence of thoughts from root to node."""
        sequence = []
        current = node
        while current:
            sequence.insert(0, current.thought)
            current = current.parent
        return sequence
    
    def search(
        self,
        initial_prompt: str,
        objective: str,
        max_steps: int = 10
    ) -> List[str]:
        """Perform tree search to find solution path.
        
        Args:
            initial_prompt: Starting prompt/context
            objective: Goal/objective
            max_steps: Maximum search steps
            
        Returns:
            Best thought sequence found
        """
        # Generate initial thoughts
        thoughts = self.generate_thoughts(initial_prompt)
        
        # Create root nodes
        nodes = [
            ThoughtNode(
                thought=t,
                value=self.estimate_value([t], objective),
                children=[],
                parent=None,
                depth=0
            )
            for t in thoughts
        ]
        
        best_sequence = None
        best_value = 0.0
        
        for _ in range(max_steps):
            # Sort nodes by value
            nodes.sort(key=lambda n: n.value, reverse=True)
            nodes = nodes[:self.beam_width]  # Keep top k
            
            # Check if we have a better sequence
            if nodes and nodes[0].value > best_value:
                best_value = nodes[0].value
                best_sequence = self._get_thought_sequence(nodes[0])
            
            # Expand nodes
            new_nodes = []
            for node in nodes:
                children = self.expand_node(node, initial_prompt, objective)
                new_nodes.extend(children)
            
            nodes = new_nodes
            if not nodes:  # No more nodes to expand
                break
                
        return best_sequence if best_sequence else []

def evaluate_tot_solution(
    model: VishwamAI,
    solution: List[str],
    objective: str
) -> Tuple[float, str]:
    """Evaluate a Tree of Thoughts solution.
    
    Args:
        model: Language model
        solution: Sequence of thoughts
        objective: Original objective
        
    Returns:
        Tuple of (score, feedback)
    """
    eval_prompt = (
        f"Objective: {objective}\n\n"
        f"Solution steps:\n"
        + "\n".join(f"{i+1}. {s}" for i, s in enumerate(solution))
        + "\n\nProvide:\n1. A score from 0 to 1\n2. Brief feedback"
    )
    
    inputs = model.tokenizer.encode(
        eval_prompt,
        return_tensors="jax",
        max_length=model.config.max_seq_len,
        truncation=True
    )
    
    outputs = model.generate(
        input_ids=inputs,
        max_length=200,
        temperature=0.3
    )
    
    response = model.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # Extract score and feedback
    try:
        score = float(next(
            float(s) for s in response.split() 
            if s.replace(".", "").isdigit()
        ))
        score = max(0.0, min(1.0, score))
    except (StopIteration, ValueError):
        score = 0.0
        
    return score, response
