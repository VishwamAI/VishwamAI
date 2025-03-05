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
    max_thought_length: int = 20  # Maximum length for generated text

class TreeOfThoughts:
    """
    Updated Tree of Thoughts implementation for enhancing reasoning with VishwamAI models.
    Generates text-based thoughts and evaluates them with task-specific relevance.
    """
    
    def __init__(
        self,
        transformer,
        tokenizer,
        max_thoughts: int = 5,
        max_depth: int = 3,
        beam_width: int = 5,
        pruning_threshold: float = 0.3,
        exploration_factor: float = 1.0,
    ):
        self.transformer = transformer
        self.tokenizer = tokenizer
        self.config = ToTConfig(
            max_thoughts=max_thoughts,
            max_depth=max_depth,
            beam_width=beam_width,
            pruning_threshold=pruning_threshold,
            exploration_factor=exploration_factor
        )
    
    def __call__(self, features: jnp.ndarray, rng_key: jnp.ndarray, prompt: str = "", search_strategy: str = "beam") -> Thought:
        """
        Generate a tree of thoughts and return the best one.
        
        Args:
            features: Input features (typically from transformer)
            rng_key: JAX PRNG key
            prompt: Initial text prompt to start thought generation
            search_strategy: The search algorithm to use
            
        Returns:
            The best thought from the search
        """
        self.config.search_strategy = search_strategy
        
        # Generate initial thoughts
        initial_thoughts = self._generate_thoughts(
            features, 
            None,  # No parent for initial thoughts
            depth=0,
            prompt=prompt,
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
    
    def _evaluate_thought(self, thought_embedding: jnp.ndarray, thought_text: str, parent_embedding: Optional[jnp.ndarray] = None) -> float:
        """
        Evaluate the quality/value of a thought with task-specific relevance.
        
        Args:
            thought_embedding: Embedding of the thought
            thought_text: Text content of the thought
            parent_embedding: Embedding of the parent thought, if any
            
        Returns:
            A score for the thought (higher is better)
        """
        # Coherence: L2 norm of embedding
        coherence = float(jnp.linalg.norm(thought_embedding))
        
        # Progress: Similarity and difference with parent
        progress = 0.0
        if parent_embedding is not None:
            dot_product = jnp.sum(thought_embedding * parent_embedding)
            parent_norm = jnp.linalg.norm(parent_embedding)
            current_norm = jnp.linalg.norm(thought_embedding)
            similarity = dot_product / (parent_norm * current_norm + 1e-8)
            progress = 0.5 * similarity + 0.5 * (1 - similarity)
        
        # Task Relevance: Simplified heuristic (e.g., length of thought as proxy for informativeness)
        # For real tasks, replace with task-specific evaluation (e.g., correctness for math)
        task_relevance = len(thought_text.split()) / self.config.max_thought_length
        
        # Combined score
        score = 0.5 * coherence + 0.3 * progress + 0.2 * task_relevance
        return float(score)
    
    def _generate_text(self, prompt: str, max_length: int, rng_key: jnp.ndarray) -> Tuple[str, jnp.ndarray]:
        """
        Generate text for a thought using the transformer model.
        
        Args:
            prompt: Initial text to start generation
            max_length: Maximum number of tokens to generate
            rng_key: JAX random key
            
        Returns:
            Tuple of generated text and its embedding
        """
        input_ids = jnp.array(self.tokenizer.encode(prompt, add_special_tokens=True))
        current_ids = input_ids
        
        for _ in range(max_length):
            rng_key, subkey = jax.random.split(rng_key)
            outputs = self.transformer(current_ids[None, :], deterministic=False, rngs={'dropout': subkey})
            next_token_logits = outputs['logits'][0, -1, :]
            next_token_probs = nn.softmax(next_token_logits / self.config.temperature, axis=-1)
            next_token = jax.random.categorical(subkey, next_token_probs)
            current_ids = jnp.concatenate([current_ids, next_token[None]], axis=0)
            if next_token == self.tokenizer.eos_id:
                break
        
        thought_text = self.tokenizer.decode(current_ids.tolist())
        outputs = self.transformer(current_ids[None, :], deterministic=True)
        thought_embedding = outputs['hidden_states'][0, -1, :]  # Last token's hidden state
        return thought_text, thought_embedding
    
    def _generate_thoughts(
        self, 
        features: jnp.ndarray,
        parent: Optional[Thought],
        depth: int,
        prompt: str,
        rng_key: jnp.ndarray,
    ) -> List[Thought]:
        """
        Generate thoughts with text content based on input features and optional parent thought.
        
        Args:
            features: Input features (not used directly here, kept for compatibility)
            parent: Optional parent thought
            depth: Current depth in the tree
            prompt: Initial text prompt
            rng_key: JAX random key
            
        Returns:
            List of generated thoughts
        """
        try:
            thoughts = []
            base_prompt = parent.content if parent else prompt
            
            for i in range(self.config.max_thoughts):
                rng_key, subkey = jax.random.split(rng_key)
                thought_text, thought_embedding = self._generate_text(
                    base_prompt,
                    self.config.max_thought_length,
                    subkey
                )
                
                # Avoid duplicate or trivial thoughts
                if thought_text == base_prompt or not thought_text.strip():
                    continue
                
                # Evaluate the thought
                parent_embedding = parent.embeddings if parent else None
                score = self._evaluate_thought(thought_embedding, thought_text, parent_embedding)
                
                # Create path
                path = parent.path + [parent] if parent else []
                
                thought = Thought(
                    content=thought_text,
                    score=score,
                    embeddings=thought_embedding,
                    children=[],
                    path=path,
                    depth=depth,
                    parent=parent
                )
                thoughts.append(thought)
            
            # Sort by score
            thoughts.sort(key=lambda t: t.score, reverse=True)
            
            # Prune thoughts
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
        
        best_score = thoughts[0].score
        pruning_threshold = best_score * self.config.pruning_threshold
        
        # Keep thoughts above threshold
        pruned_thoughts = [t for t in thoughts if t.score >= pruning_threshold]
        
        # Ensure diversity by checking cosine similarity
        final_thoughts = []
        for thought in pruned_thoughts:
            if not final_thoughts or all(
                jnp.sum(thought.embeddings * ft.embeddings) / (
                    jnp.linalg.norm(thought.embeddings) * jnp.linalg.norm(ft.embeddings) + 1e-8
                ) < 0.9 for ft in final_thoughts
            ):
                final_thoughts.append(thought)
            if len(final_thoughts) >= self.config.max_thoughts:
                break
        
        return final_thoughts
    
    def _beam_search(self, initial_thoughts: List[Thought], rng_key: jnp.ndarray) -> Thought:
        """
        Perform beam search with length normalization through the tree of thoughts.
        
        Args:
            initial_thoughts: Initial thoughts
            rng_key: JAX random key
            
        Returns:
            Best thought found
        """
        logger.info("Starting beam search in Tree of Thoughts")
        
        beam = initial_thoughts[:self.config.beam_width]
        best_thought = max(initial_thoughts, key=lambda t: t.score) if initial_thoughts else None
        
        for depth in range(1, self.config.max_depth + 1):
            new_beam = []
            for thought in beam:
                rng_key, child_key = jax.random.split(rng_key)
                children = self._generate_thoughts(
                    thought.embeddings, 
                    thought, 
                    depth, 
                    thought.content, 
                    child_key
                )
                thought_with_children = thought._replace(children=children)
                for child in children:
                    # Length-normalized score
                    length = len(child.content.split())
                    normalized_score = child.score / (length ** 0.5) if length > 0 else child.score
                    new_beam.append(child._replace(score=normalized_score))
            
            if not new_beam:
                break
                
            new_beam.sort(key=lambda t: t.score, reverse=True)
            beam = new_beam[:self.config.beam_width]
            
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
        
        def dfs_recursive(thought, depth, best_found, rng_key):
            if depth >= self.config.max_depth:
                return best_found
            
            rng_key, child_key = jax.random.split(rng_key)
            children = self._generate_thoughts(thought.embeddings, thought, depth + 1, thought.content, child_key)
            thought_with_children = thought._replace(children=children)
            
            for child in children:
                if child.score > best_found.score:
                    best_found = child
                rng_key, explore_key = jax.random.split(rng_key)
                best_found = dfs_recursive(child, depth + 1, best_found, explore_key)
            
            return best_found
        
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
        queue = [(thought, 0) for thought in initial_thoughts]
        
        while queue:
            thought, depth = queue.pop(0)
            if thought.score > best_thought.score:
                best_thought = thought
                
            if depth >= self.config.max_depth:
                continue
                
            rng_key, child_key = jax.random.split(rng_key)
            children = self._generate_thoughts(thought.embeddings, thought, depth + 1, thought.content, child_key)
            thought_with_children = thought._replace(children=children)
            queue.extend([(child, depth + 1) for child in children])
        
        return best_thought
    
    def _monte_carlo_tree_search(self, initial_thoughts: List[Thought], rng_key: jnp.ndarray) -> Thought:
        """
        Perform Monte Carlo Tree Search through the tree of thoughts (basic implementation).
        
        Args:
            initial_thoughts: Initial thoughts
            rng_key: JAX random key
            
        Returns:
            Best thought found
        """
        logger.info("Starting Monte Carlo Tree Search in Tree of Thoughts")
        
        best_thought = max(initial_thoughts, key=lambda t: t.score) if initial_thoughts else None
        visits = {thought.content: 0 for thought in initial_thoughts}
        scores = {thought.content: thought.score for thought in initial_thoughts}
        
        def select_node(thoughts, visits, scores):
            # Simple UCT-like selection
            return max(thoughts, key=lambda t: scores[t.content] + self.config.exploration_factor * np.sqrt(np.log(sum(visits.values()) + 1) / (visits[t.content] + 1e-6)))
        
        for _ in range(self.config.max_thoughts * self.config.max_depth):  # Limited iterations
            current = select_node(initial_thoughts, visits, scores)
            depth = 0
            while depth < self.config.max_depth:
                rng_key, child_key = jax.random.split(rng_key)
                children = self._generate_thoughts(current.embeddings, current, depth + 1, current.content, child_key)
                if not children:
                    break
                current = max(children, key=lambda t: t.score)
                visits[current.content] = visits.get(current.content, 0) + 1
                scores[current.content] = scores.get(current.content, current.score)
                if current.score > best_thought.score:
                    best_thought = current
                depth += 1
        
        return best_thought
