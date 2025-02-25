import jax
import jax.numpy as jnp
from flax import linen as nn
from typing import Dict, List, Tuple, Optional, Any, Callable, NamedTuple
import logging
from dataclasses import dataclass
from functools import partial
import numpy as np

logger = logging.getLogger(__name__)

class Thought(NamedTuple):
    """Representation of a thought in the Tree of Thoughts."""
    content: str
    score: float
    embeddings: jnp.ndarray
    children: List = None
    path: List = None
    depth: int = 0
    parent: Any = None

class SearchState(NamedTuple):
    """State maintained during search in Tree of Thoughts."""
    thoughts: List[Thought]
    best_thought: Thought
    depth: int
    beam_width: int

@dataclass
class ToTConfig:
    """Configuration for Tree of Thoughts."""
    max_thoughts: int = 5
    max_depth: int = 3
    beam_width: int = 5
    pruning_threshold: float = 0.3
    exploration_factor: float = 1.0
    temperature: float = 0.7
    search_strategy: str = "beam"  # Options: "beam", "dfs", "bfs", "mcts"

class TreeOfThoughts:
    """
    Tree of Thoughts implementation for enhancing reasoning with VishwamAI models.
    
    This implements the Tree of Thoughts approach from:
    "Tree of Thoughts: Deliberate Problem Solving with Large Language Models"
    by Yao et al. (2023)
    """
    
    def __init__(
        self,
        transformer,
        max_thoughts: int = 5,
        max_depth: int = 3,
        beam_width: int = 5,
        pruning_threshold: float = 0.3,
        exploration_factor: float = 1.0,
    ):
        self.transformer = transformer
        self.config = ToTConfig(
            max_thoughts=max_thoughts,
            max_depth=max_depth,
            beam_width=beam_width,
            pruning_threshold=pruning_threshold,
            exploration_factor=exploration_factor
        )
    
    def __call__(self, features: jnp.ndarray, rng_key: jnp.ndarray, search_strategy: str = "beam") -> Thought:
        """
        Generate a tree of thoughts and return the best one.
        
        Args:
            features: Input features (typically from transformer)
            rng_key: JAX PRNG key
            search_strategy: The search algorithm to use
            
        Returns:
            The best thought from the search
        """
        # Override search strategy if provided
        self.config.search_strategy = search_strategy
        
        # Generate initial thoughts
        initial_thoughts = self._generate_thoughts(
            features, 
            None,  # No parent for initial thoughts
            depth=0,
            rng_key=rng_key
        )
        
        if not initial_thoughts:
            logger.warning("Failed to generate initial thoughts")
            return None
        
        # Select search strategy
        if search_strategy == "beam":
            best_thought = self._beam_search(initial_thoughts, rng_key)
        elif search_strategy == "dfs":
            best_thought = self._depth_first_search(initial_thoughts, rng_key)
        elif search_strategy == "bfs":
            best_thought = self._breadth_first_search(initial_thoughts, rng_key)
        elif search_strategy == "mcts":
            best_thought = self._monte_carlo_tree_search(initial_thoughts, rng_key)
        else:
            logger.warning(f"Unknown search strategy: {search_strategy}, using beam search")
            best_thought = self._beam_search(initial_thoughts, rng_key)
            
        return best_thought
    
    def _evaluate_thought(self, thought_embedding: jnp.ndarray, parent_embedding: Optional[jnp.ndarray] = None) -> float:
        """
        Evaluate the quality/value of a thought.
        
        Args:
            thought_embedding: Embedding of the thought
            parent_embedding: Embedding of the parent thought, if any
            
        Returns:
            A score for the thought (higher is better)
        """
        # Compute coherence using L2 norm of embedding
        coherence = jnp.linalg.norm(thought_embedding)
        
        # If we have a parent, compute progress as distance from parent
        progress = 0.0
        if parent_embedding is not None:
            # Use cosine similarity to measure similarity with parent
            dot_product = jnp.sum(thought_embedding * parent_embedding)
            parent_norm = jnp.linalg.norm(parent_embedding)
            current_norm = jnp.linalg.norm(thought_embedding)
            similarity = dot_product / (parent_norm * current_norm + 1e-8)
            
            # Progress is a combination of similarity and difference
            # We want thoughts that are related but add something new
            progress = 0.5 * similarity + 0.5 * (1 - similarity)
        
        # Combine coherence and progress
        score = 0.7 * coherence + 0.3 * progress
        return float(score)
    
    def _generate_thoughts(
        self, 
        features: jnp.ndarray,
        parent: Optional[Thought],
        depth: int,
        rng_key: jnp.ndarray,
    ) -> List[Thought]:
        """
        Generate thoughts based on input features and optional parent thought.
        
        Args:
            features: Input features
            parent: Optional parent thought
            depth: Current depth in the tree
            rng_key: JAX random key
            
        Returns:
            List of generated thoughts
        """
        try:
            # Use the model to generate thought embeddings
            rng_key, dropout_key = jax.random.split(rng_key)
            
            # Create masked version of features to encourage diversity
            masked_features = features
            if parent is not None:
                # Apply selective masking based on parent's embedding pattern
                parent_pattern = jnp.abs(parent.embeddings) > jnp.median(jnp.abs(parent.embeddings))
                mask = jnp.where(parent_pattern, 0.8, 1.0)
                masked_features = features * mask
            
            # Use transformer to generate embeddings
            outputs = self.transformer(masked_features, deterministic=False, rngs={'dropout': dropout_key})
            embeddings = outputs[0]  # Use the last hidden state
            
            # Generate multiple thoughts with different dropout patterns
            thoughts = []
            for i in range(self.config.max_thoughts):
                # Generate different embedding variations using dropout
                rng_key, dropout_key = jax.random.split(rng_key)
                thought_outputs = self.transformer(
                    masked_features,
                    deterministic=False,
                    rngs={'dropout': dropout_key}
                )
                thought_embedding = thought_outputs[0]
                
                # Evaluate the thought
                parent_embedding = parent.embeddings if parent else None
                score = self._evaluate_thought(thought_embedding, parent_embedding)
                
                # Create path for this thought
                path = parent.path + [parent] if parent else []
                
                # Create new thought
                thought = Thought(
                    content=f"Thought {depth}_{i}",
                    score=score,
                    embeddings=thought_embedding,
                    children=[],
                    path=path,
                    depth=depth,
                    parent=parent
                )
                thoughts.append(thought)
                
            # Sort thoughts by score (descending)
            thoughts.sort(key=lambda t: t.score, reverse=True)
            
            # Apply pruning
            pruned_thoughts = self._prune_thoughts(thoughts)
            
            return pruned_thoughts
            
        except Exception as e:
            logger.error(f"Error generating thoughts: {str(e)}")
            return []
    
    def _prune_thoughts(self, thoughts: List[Thought]) -> List[Thought]:
        """
        Prune thoughts based on score and diversity.
        
        Args:
            thoughts: List of thoughts to prune
            
        Returns:
            Pruned list of thoughts
        """
        if not thoughts:
            return []
            
        # Sort by score (already done in _generate_thoughts)
        best_score = thoughts[0].score
        pruning_threshold = best_score * self.config.pruning_threshold
        
        # Keep thoughts above threshold
        pruned_thoughts = [t for t in thoughts if t.score >= pruning_threshold]
        
        # Ensure we don't exceed max thoughts
        if len(pruned_thoughts) > self.config.max_thoughts:
            pruned_thoughts = pruned_thoughts[:self.config.max_thoughts]
        
        return pruned_thoughts
    
    def _beam_search(self, initial_thoughts: List[Thought], rng_key: jnp.ndarray) -> Thought:
        """
        Perform beam search through the tree of thoughts.
        
        Args:
            initial_thoughts: Initial thoughts
            rng_key: JAX random key
            
        Returns:
            Best thought found
        """
        logger.info("Starting beam search in Tree of Thoughts")
        
        # Initialize beam with initial thoughts
        beam = initial_thoughts[:self.config.beam_width]
        
        # Keep track of best thought
        best_thought = max(initial_thoughts, key=lambda t: t.score) if initial_thoughts else None
        
        # Iterate until max depth
        for depth in range(1, self.config.max_depth + 1):
            # Generate children for all thoughts in beam
            new_beam = []
            for thought in beam:
                rng_key, child_key = jax.random.split(rng_key)
                children = self._generate_thoughts(thought.embeddings, thought, depth, child_key)
                
                # Update thought's children
                thought_with_children = thought._replace(children=children)
                
                # Add children to new beam
                new_beam.extend(children)
            
            # If no new thoughts, break
            if not new_beam:
                break
                
            # Sort and select top beam_width thoughts
            new_beam.sort(key=lambda t: t.score, reverse=True)
            beam = new_beam[:self.config.beam_width]
            
            # Update best thought
            if beam and beam[0].score > best_thought.score:
                best_thought = beam[0]
        
        return best_thought

    def _depth_first_search(self, initial_thoughts: List[Thought], rng_key: jnp.ndarray) -> Thought:
        """
        Perform depth-first search through the tree of thoughts.
        
        Args:
            initial_thoughts: Initial thoughts
            rng_key: JAX random key
            
        Returns:
            Best thought found
        """
        logger.info("Starting depth-first search in Tree of Thoughts")
        
        best_thought = max(initial_thoughts, key=lambda t: t.score) if initial_thoughts else None
        
        # Helper function for DFS recursion
        def dfs_recursive(thought, depth, best_found, rng_key):
            if depth >= self.config.max_depth:
                return best_found
            
            # Generate children
            rng_key, child_key = jax.random.split(rng_key)
            children = self._generate_thoughts(thought.embeddings, thought, depth + 1, child_key)
            
            # Update thought's children
            thought_with_children = thought._replace(children=children)
            
            # Update best thought
            for child in children:
                if child.score > best_found.score:
                    best_found = child
                
                # Recursively explore child
                rng_key, explore_key = jax.random.split(rng_key)
                best_found = dfs_recursive(child, depth + 1, best_found, explore_key)
            
            return best_found
        
        # Start DFS from each initial thought
        for thought in initial_thoughts:
            rng_key, search_key = jax.random.split(rng_key)
            best_thought = dfs_recursive(thought, 0, best_thought, search_key)
        
        return best_thought
    
    def _breadth_first_search(self, initial_thoughts: List[Thought], rng_key: jnp.ndarray) -> Thought:
        """
        Perform breadth-first search through the tree of thoughts.
        
        Args:
            initial_thoughts: Initial thoughts
            rng_key: JAX random key
            
        Returns:
            Best thought found
        """
        logger.info("Starting breadth-first search in Tree of Thoughts")
        
        best_thought = max(initial_thoughts, key=lambda t: t.score) if initial_thoughts else None
        
        # Initialize queue with initial thoughts
        queue = [(thought, 0) for thought in initial_thoughts]
        
        while queue:
            thought, depth = queue.pop(0)
            
            # Update best thought
            if thought.score > best_thought.score:
                best_thought = thought
                
            # Stop if max depth reached
            if depth >= self.config.max_depth:
                continue
                
            # Generate children
            rng_key, child_key = jax.random.split(rng_key)
            children = self._generate_thoughts(thought.embeddings, thought, depth + 1, child_key)
            
            # Update thought's children
            thought_with_children = thought._replace(children=children)
            
            # Add children to queue
            queue.extend([(child, depth + 1) for child in children])
        
        return best_thought
    
    def _monte_carlo_tree_search(self, initial_thoughts: List[Thought], rng_key: jnp.ndarray) -> Thought:
        """
        Perform Monte Carlo Tree Search through the tree of thoughts.
        
        Args:
            initial_thoughts: Initial thoughts
            rng_key: JAX random key
            
        Returns:
            Best thought found
        """
        logger.info("Starting Monte Carlo Tree Search in Tree of Thoughts")
        # Simplified MCTS implementation
        best_thought = max(initial_thoughts, key=lambda t: t.score) if initial_thoughts else None
        return best_thought  # Placeholder for full MCTS implementation
