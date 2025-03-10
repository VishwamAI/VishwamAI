"""
Tree of Thoughts (ToT) implementation for VishwamAI transformer.
Enables sophisticated multi-path reasoning with branching thought processes.
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from queue import PriorityQueue
import numpy as np
from .transformer import EnhancedTransformerModel
from .cot import format_cot_prompt, extract_reasoning_steps

@dataclass
class ThoughtNode:
    """Represents a node in the thought tree."""
    thought: str
    score: float
    parent: Optional['ThoughtNode'] = None
    children: List['ThoughtNode'] = None
    depth: int = 0
    
    def __post_init__(self):
        if self.children is None:
            self.children = []
    
    def __lt__(self, other):
        return self.score > other.score  # Higher scores have priority

class TreeOfThoughts:
    """
    Implements Tree of Thoughts reasoning with beam search and pruning.
    """
    
    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        temperature: float = 0.7,
        max_depth: int = 5,
        beam_width: int = 3,
        num_samples: int = 5,
        max_length: int = 512
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.temperature = temperature
        self.max_depth = max_depth
        self.beam_width = beam_width
        self.num_samples = num_samples
        self.max_length = max_length
    
    def evaluate_thought(
        self,
        thought: str,
        context: str,
        criteria: Dict[str, float]
    ) -> float:
        """
        Evaluate a thought based on multiple criteria.
        
        Args:
            thought: The thought to evaluate
            context: Context including previous thoughts
            criteria: Dictionary of evaluation criteria and weights
        """
        score = 0.0
        total_weight = sum(criteria.values())
        
        # Evaluate coherence
        if 'coherence' in criteria:
            # Check for logical connectors and flow
            coherence_score = sum(1 for connector in 
                ['because', 'therefore', 'thus', 'so', 'hence']
                if connector in thought.lower())
            score += criteria['coherence'] * (coherence_score / 3)  # Normalize
        
        # Evaluate relevance to context
        if 'relevance' in criteria:
            # Simple keyword matching for now
            context_keywords = set(context.lower().split())
            thought_keywords = set(thought.lower().split())
            overlap = len(context_keywords.intersection(thought_keywords))
            relevance_score = overlap / max(len(context_keywords), 1)
            score += criteria['relevance'] * relevance_score
        
        # Evaluate specificity
        if 'specificity' in criteria:
            # Count specific details (numbers, named entities, etc.)
            specifics = sum(c.isdigit() or c == '$' for c in thought)
            specificity_score = min(specifics / 10, 1.0)  # Normalize
            score += criteria['specificity'] * specificity_score
        
        return score / total_weight
    
    def generate_thoughts(
        self,
        prompt: str,
        context: str,
        num_samples: Optional[int] = None
    ) -> List[str]:
        """
        Generate multiple possible next thoughts.
        
        Args:
            prompt: The current prompt
            context: Previous context
            num_samples: Number of thoughts to generate
        """
        if num_samples is None:
            num_samples = self.num_samples
            
        thoughts = []
        for _ in range(num_samples):
            # Generate next thought
            input_ids = self.tokenizer.encode(prompt + context)
            output = self.model.generate(
                input_ids,
                max_length=self.max_length,
                temperature=self.temperature,
                top_p=0.95
            )
            decoded = self.tokenizer.decode(output[0])
            
            # Extract the new thought
            new_thought = decoded[len(prompt + context):].strip()
            thoughts.append(new_thought)
            
        return thoughts
    
    def build_thought_tree(
        self,
        question: str,
        evaluation_criteria: Optional[Dict[str, float]] = None
    ) -> ThoughtNode:
        """
        Build a tree of thoughts using beam search.
        
        Args:
            question: The question to reason about
            evaluation_criteria: Criteria for evaluating thoughts
        """
        if evaluation_criteria is None:
            evaluation_criteria = {
                'coherence': 0.4,
                'relevance': 0.4,
                'specificity': 0.2
            }
            
        # Initialize root
        root_prompt = format_cot_prompt(question)
        root_thoughts = self.generate_thoughts(root_prompt, "")
        
        # Create root nodes
        root_nodes = []
        for thought in root_thoughts:
            score = self.evaluate_thought(thought, "", evaluation_criteria)
            root_nodes.append(ThoughtNode(thought=thought, score=score))
        
        # Sort and keep top-k
        root_nodes.sort(reverse=True, key=lambda x: x.score)
        beam = root_nodes[:self.beam_width]
        
        # Build tree level by level
        for depth in range(1, self.max_depth):
            candidates = []
            
            # Generate children for each node in beam
            for parent in beam:
                context = self._get_thought_path(parent)
                prompt = root_prompt + "\n".join(context) + "\n"
                
                # Generate and evaluate children
                child_thoughts = self.generate_thoughts(prompt, context[-1])
                for thought in child_thoughts:
                    score = self.evaluate_thought(
                        thought,
                        "\n".join(context),
                        evaluation_criteria
                    )
                    child = ThoughtNode(
                        thought=thought,
                        score=score,
                        parent=parent,
                        depth=depth
                    )
                    candidates.append(child)
                    parent.children.append(child)
            
            # Update beam with top-k candidates
            candidates.sort(reverse=True, key=lambda x: x.score)
            beam = candidates[:self.beam_width]
            
            # Early stopping if no good candidates
            if not beam or beam[0].score < 0.2:
                break
        
        return root_nodes[0]  # Return root of best tree
    
    def _get_thought_path(self, node: ThoughtNode) -> List[str]:
        """Get the path of thoughts from root to node."""
        path = []
        current = node
        while current:
            path.append(current.thought)
            current = current.parent
        return path[::-1]
    
    def search_best_path(
        self,
        root: ThoughtNode,
        evaluation_criteria: Optional[Dict[str, float]] = None
    ) -> List[str]:
        """
        Search for the best path in the thought tree.
        
        Args:
            root: Root node of the thought tree
            evaluation_criteria: Criteria for path evaluation
        """
        if evaluation_criteria is None:
            evaluation_criteria = {
                'coherence': 0.4,
                'relevance': 0.4,
                'specificity': 0.2
            }
            
        # Use priority queue for best-first search
        queue = PriorityQueue()
        queue.put((1.0, root))  # Start with root
        visited = set()
        best_path = None
        best_score = float('-inf')
        
        while not queue.empty():
            score, node = queue.get()
            
            # Get current path
            current_path = self._get_thought_path(node)
            path_key = tuple(current_path)
            
            if path_key in visited:
                continue
                
            visited.add(path_key)
            
            # Evaluate complete path
            if len(current_path) > 1:  # At least 2 thoughts
                path_score = self.evaluate_thought(
                    " ".join(current_path),
                    current_path[0],  # Use first thought as context
                    evaluation_criteria
                )
                
                if path_score > best_score:
                    best_score = path_score
                    best_path = current_path
            
            # Add children to queue
            for child in node.children:
                if tuple(self._get_thought_path(child)) not in visited:
                    queue.put((child.score, child))
        
        return best_path or self._get_thought_path(root)
    
    def reason(
        self,
        question: str,
        evaluation_criteria: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Perform tree of thoughts reasoning.
        
        Args:
            question: Question to reason about
            evaluation_criteria: Criteria for evaluating thoughts
        """
        # Build thought tree
        root = self.build_thought_tree(question, evaluation_criteria)
        
        # Find best reasoning path
        best_path = self.search_best_path(root, evaluation_criteria)
        
        # Extract final answer
        reasoning = "\n".join(best_path)
        steps, answer = extract_reasoning_steps(reasoning)
        
        return {
            'reasoning_steps': steps,
            'answer': answer,
            'thought_tree': root,
            'best_path': best_path,
            'full_reasoning': reasoning
        }

def evaluate_tot_solution(
    reasoning_output: Dict[str, Any],
    criteria: Optional[Dict[str, float]] = None
) -> Dict[str, float]:
    """
    Evaluate a ToT reasoning solution.
    
    Args:
        reasoning_output: Output from ToT reasoning
        criteria: Evaluation criteria and weights
    """
    if criteria is None:
        criteria = {
            'path_length': 0.3,
            'reasoning_depth': 0.4,
            'answer_clarity': 0.3
        }
        
    scores = {}
    
    # Evaluate path length
    if 'path_length' in criteria:
        optimal_length = 5
        path_length = len(reasoning_output['reasoning_steps'])
        scores['path_length'] = (
            criteria['path_length'] *
            max(0, 1 - abs(path_length - optimal_length) / optimal_length)
        )
    
    # Evaluate reasoning depth
    if 'reasoning_depth' in criteria:
        def count_branches(node: ThoughtNode) -> int:
            if not node.children:
                return 1
            return 1 + sum(count_branches(child) for child in node.children)
        
        total_branches = count_branches(reasoning_output['thought_tree'])
        scores['reasoning_depth'] = (
            criteria['reasoning_depth'] *
            min(total_branches / 10, 1.0)  # Normalize to [0, 1]
        )
    
    # Evaluate answer clarity
    if 'answer_clarity' in criteria:
        answer = reasoning_output['answer']
        # Simple heuristics for answer clarity
        has_clear_statement = any(
            marker in answer.lower()
            for marker in ['therefore', 'thus', 'conclusion', 'answer is']
        )
        has_supporting_evidence = any(
            marker in answer.lower()
            for marker in ['because', 'since', 'as shown by']
        )
        
        clarity_score = (has_clear_statement + has_supporting_evidence) / 2
        scores['answer_clarity'] = criteria['answer_clarity'] * clarity_score
    
    # Calculate total score
    scores['total'] = sum(scores.values())
    
    return scores