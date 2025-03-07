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
    device_mesh: Optional[Any] = None  # TPU device mesh for parallel processing

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
    use_bfloat16: bool = True  # Use TPU-optimized bfloat16 precision
    batch_size_per_device: int = 2  # Batch size per TPU device 
    parallel_thoughts: int = 8  # Number of thoughts to process in parallel
    chunk_size: int = 32  # Chunk size for TPU memory optimization

class TreeOfThoughts:
    """
    TPU-optimized Tree of Thoughts implementation for enhancing reasoning with VishwamAI models.
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
        use_tpu: bool = True,
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
        
        # TPU-specific initialization
        if use_tpu:
            self._setup_tpu_devices()
        else:
            self.device_mesh = None
            logger.info("TPU optimization disabled for ToT")
    
    def _setup_tpu_devices(self):
        """Configure TPU devices for optimal ToT processing."""
        try:
            # Get available TPU devices
            devices = jax.devices("tpu")
            num_devices = len(devices)
            
            if num_devices > 0:
                # Create a device mesh for sharding
                device_mesh = jax.sharding.Mesh(
                    np.array(devices).reshape(-1),
                    ('batch',)
                )
                self.device_mesh = device_mesh
                
                # Adjust batch size based on device count
                self.config.parallel_thoughts = min(self.config.parallel_thoughts, num_devices)
                logger.info(f"ToT configured with {num_devices} TPU devices")
            else:
                self.device_mesh = None
                logger.warning("No TPU devices found, falling back to CPU")
        except Exception as e:
            self.device_mesh = None
            logger.error(f"Failed to initialize TPU for ToT: {str(e)}")
    
    @partial(jax.jit, static_argnums=(0,))
    def _batch_thought_generation(self, prompts, rng_keys):
        """TPU-optimized batched thought generation."""
        # This would be implemented to process multiple thoughts in parallel on TPU
        batch_size = len(prompts)
        if batch_size == 0:
            return []
        
        # Convert to optimal dtype for TPU
        input_array = self.tokenizer.batch_encode(prompts)
        if self.config.use_bfloat16:
            model_dtype = jnp.bfloat16
        else:
            model_dtype = jnp.float32
        
        # Process in chunks to optimize memory
        chunked_outputs = []
        for i in range(0, input_array.shape[0], self.config.chunk_size):
            chunk = input_array[i:i+self.config.chunk_size]
            
            # Forward pass with TPU optimization
            outputs = self.transformer(
                chunk, 
                deterministic=False, 
                rngs={'dropout': rng_keys[i:i+self.config.chunk_size]},
                output_hidden_states=True
            )
            
            chunked_outputs.append({
                'logits': outputs['logits'].astype(model_dtype),
                'hidden_states': outputs['hidden_states'].astype(model_dtype)
            })
        
        # Combine chunk results
        combined_outputs = jax.tree_map(
            lambda *xs: jnp.concatenate(xs, axis=0),
            *chunked_outputs
        )
        
        return combined_outputs
    
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
        
        # Generate initial thoughts with TPU optimization
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
    
    @partial(jax.jit, static_argnums=(0,))
    def _evaluate_thought(self, thought_embedding: jnp.ndarray, thought_text: str, parent_embedding: Optional[jnp.ndarray] = None) -> float:
        """
        TPU-optimized thought quality evaluation.
        
        Args:
            thought_embedding: Embedding of the thought
            thought_text: Text content of the thought
            parent_embedding: Embedding of the parent thought, if any
            
        Returns:
            A score for the thought (higher is better)
        """
        # Coherence: L2 norm of embedding (TPU-optimized)
        coherence = jnp.linalg.norm(thought_embedding)
        
        # Progress: Similarity and difference with parent (TPU-optimized)
        progress = 0.0
        if parent_embedding is not None:
            dot_product = jnp.sum(thought_embedding * parent_embedding)
            parent_norm = jnp.linalg.norm(parent_embedding)
            current_norm = jnp.linalg.norm(thought_embedding)
            similarity = dot_product / (parent_norm * current_norm + 1e-8)
            progress = 0.5 * similarity + 0.5 * (1 - similarity)
        
        # Task Relevance: Simplified heuristic (e.g., length of thought as proxy for informativeness)
        task_relevance = jnp.minimum(
            len(thought_text.split()) / self.config.max_thought_length,
            1.0
        )
        
        # Combined score (TPU-optimized)
        score = 0.5 * coherence + 0.3 * progress + 0.2 * task_relevance
        return float(score)
    
    @partial(jax.jit, static_argnums=(0, 2))
    def _generate_text(self, prompt: str, max_length: int, rng_key: jnp.ndarray) -> Tuple[str, jnp.ndarray]:
        """
        TPU-optimized text generation for thoughts.
        
        Args:
            prompt: Initial text to start generation
            max_length: Maximum number of tokens to generate
            rng_key: JAX random key
            
        Returns:
            Tuple of generated text and its embedding
        """
        input_ids = jnp.array(self.tokenizer.encode(prompt, add_special_tokens=True))
        current_ids = input_ids
        
        # Store generated tokens
        all_tokens = [current_ids]
        
        # Use scan for better TPU performance
        def generation_step(carry, _):
            current_ids, rng = carry
            rng, subkey = jax.random.split(rng)
            
            # Run model with bfloat16 precision for TPU
            outputs = self.transformer(
                current_ids[None, :],
                deterministic=False,
                rngs={'dropout': subkey},
                output_hidden_states=True
            )
            
            next_token_logits = outputs['logits'][0, -1, :]
            next_token_probs = jax.nn.softmax(
                next_token_logits / self.config.temperature,
                axis=-1
            )
            
            # Sample token
            next_token = jax.random.categorical(subkey, next_token_probs)
            new_ids = jnp.concatenate([current_ids, next_token[None]], axis=0)
            
            # Check if we should terminate
            done = (next_token == self.tokenizer.eos_id)
            
            return (new_ids, rng), (new_ids, done)
        
        # Run generation with TPU-optimized scan
        (final_ids, _), (all_new_ids, dones) = jax.lax.scan(
            generation_step,
            (current_ids, rng_key),
            None,
            length=max_length
        )
        
        # Remove padding and get final output
        thought_text = self.tokenizer.decode(final_ids.tolist())
        
        # Get embedding from the transformer
        outputs = self.transformer(final_ids[None, :], deterministic=True, output_hidden_states=True)
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
        TPU-optimized parallel thought generation.
        
        Args:
            features: Input features
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
            
            # Split into batches for TPU parallel processing
            batch_size = min(self.config.parallel_thoughts, self.config.max_thoughts)
            num_batches = (self.config.max_thoughts + batch_size - 1) // batch_size
            
            for batch_idx in range(num_batches):
                # Generate batch of random keys
                batch_keys = jax.random.split(rng_key, batch_size)
                rng_key = batch_keys[0]  # Update main key
                
                # Process batch in parallel on TPU when possible
                if self.device_mesh is not None:
                    with jax.sharding.Mesh(self.device_mesh, ('batch',)):
                        batch_results = []
                        for i in range(min(batch_size, self.config.max_thoughts - batch_idx * batch_size)):
                            subkey = batch_keys[i]
                            thought_text, thought_embedding = self._generate_text(
                                base_prompt,
                                self.config.max_thought_length,
                                subkey
                            )
                            
                            # Skip invalid thoughts
                            if thought_text == base_prompt or not thought_text.strip():
                                continue
                            
                            # Evaluate thought with TPU optimization
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
                            batch_results.append(thought)
                            
                        thoughts.extend(batch_results)
                else:
                    # Fallback to sequential processing
                    for i in range(min(batch_size, self.config.max_thoughts - batch_idx * batch_size)):
                        subkey = batch_keys[i]
                        thought_text, thought_embedding = self._generate_text(
                            base_prompt,
                            self.config.max_thought_length,
                            subkey
                        )
                        
                        # Skip invalid thoughts
                        if thought_text == base_prompt or not thought_text.strip():
                            continue
                        
                        # Evaluate thought
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
        Prune thoughts based on score and diversity with TPU optimizations.
        
        Args:
            thoughts: List of thoughts to prune
            
        Returns:
            Pruned list of thoughts
        """
        if not thoughts:
            return []
        
        best_score = thoughts[0].score
        pruning_threshold = best_score * self.config.pruning_threshold
        
        # Keep thoughts above threshold (TPU-optimized when possible)
        pruned_thoughts = []
        for thought in thoughts:
            if thought.score >= pruning_threshold:
                pruned_thoughts.append(thought)
        
        # Ensure diversity by checking cosine similarity
        final_thoughts = []
        for thought in pruned_thoughts:
            # Calculate similarities in parallel when possible
            similarity_check = True
            for ft in final_thoughts:
                # TPU-optimized cosine similarity
                similarity = jnp.sum(thought.embeddings * ft.embeddings) / (
                    jnp.linalg.norm(thought.embeddings) * jnp.linalg.norm(ft.embeddings) + 1e-8
                )
                if similarity >= 0.9:
                    similarity_check = False
                    break
            
            if similarity_check:
                final_thoughts.append(thought)
            
            if len(final_thoughts) >= self.config.max_thoughts:
                break
        
        return final_thoughts
    
    @partial(jax.jit, static_argnums=(0,))
    def _compute_normalized_scores(self, thoughts: List[Thought]) -> List[float]:
        """TPU-optimized length-normalized scoring."""
        scores = []
        for thought in thoughts:
            length = len(thought.content.split())
            normalized_score = thought.score / (length ** 0.5) if length > 0 else thought.score
            scores.append(normalized_score)
        return scores
    
    def _beam_search(self, initial_thoughts: List[Thought], rng_key: jnp.ndarray) -> Thought:
        """
        Perform beam search with TPU optimization.
        
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
            
            # Process beam elements in batches for TPU
            for batch_start in range(0, len(beam), self.config.batch_size_per_device):
                batch_thoughts = beam[batch_start:batch_start + self.config.batch_size_per_device]
                
                for thought in batch_thoughts:
                    rng_key, child_key = jax.random.split(rng_key)
                    # Generate children with TPU optimization
                    children = self._generate_thoughts(
                        thought.embeddings, 
                        thought, 
                        depth, 
                        thought.content, 
                        child_key
                    )
                    
                    # Calculate normalized scores efficiently
                    for child in children:
                        length = len(child.content.split())
                        normalized_score = child.score / (length ** 0.5) if length > 0 else child.score
                        new_beam.append(child._replace(score=normalized_score))
            
            if not new_beam:
                break
            
            # Sort efficiently using numpy for large beams
            if len(new_beam) > 1000:
                scores = np.array([t.score for t in new_beam])
                sorted_indices = np.argsort(scores)[::-1]
                new_beam = [new_beam[i] for i in sorted_indices]
            else:
                new_beam.sort(key=lambda t: t.score, reverse=True)
            
            beam = new_beam[:self.config.beam_width]
            
            if beam and beam[0].score > best_thought.score:
                best_thought = beam[0]
        
        return best_thought
    
    def _depth_first_search(self, initial_thoughts: List[Thought], rng_key: jnp.ndarray) -> Thought:
        """
        Perform depth-first search with TPU optimization.
        
        Args:
            initial_thoughts: Initial thoughts
            rng_key: JAX random key
            
        Returns:
            Best thought found
        """
        logger.info("Starting depth-first search in Tree of Thoughts")
        
        best_thought = max(initial_thoughts, key=lambda t: t.score) if initial_thoughts else None
        
        # Use a stack-based approach instead of recursion for TPU compatibility
        stack = [(thought, 0, rng_key) for thought in initial_thoughts]
        
        while stack:
            thought, depth, current_rng = stack.pop()
            
            if thought.score > best_thought.score:
                best_thought = thought
            
            if depth >= self.config.max_depth:
                continue
            
            # Split random key for this expansion
            current_rng, child_key = jax.random.split(current_rng)
            
            # Generate children with TPU optimization
            children = self._generate_thoughts(
                thought.embeddings,
                thought,
                depth + 1,
                thought.content,
                child_key
            )
            
            # Add children to stack in reverse order (highest score first for DFS)
            children.sort(key=lambda t: t.score, reverse=True)
            for child in children:
                # Generate a new random key for each child
                current_rng, next_key = jax.random.split(current_rng)
                stack.append((child, depth + 1, next_key))
        
        return best_thought
    
    def _breadth_first_search(self, initial_thoughts: List[Thought], rng_key: jnp.ndarray) -> Thought:
        """
        Perform breadth-first search with TPU optimization.
        
        Args:
            initial_thoughts: Initial thoughts
            rng_key: JAX random key
            
        Returns:
            Best thought found
        """
        logger.info("Starting breadth-first search in Tree of Thoughts")
        
        best_thought = max(initial_thoughts, key=lambda t: t.score) if initial_thoughts else None
        queue = [(thought, 0, i) for i, thought in enumerate(initial_thoughts)]  # Include index for key splitting
        
        # Pre-allocate random keys for better TPU performance
        max_nodes = self.config.max_thoughts * self.config.max_depth * self.config.beam_width
        all_keys = jax.random.split(rng_key, max_nodes)
        key_idx = 0
        
        while queue:
            thought, depth, idx = queue.pop(0)
            
            if thought.score > best_thought.score:
                best_thought = thought
                
            if depth >= self.config.max_depth:
                continue
            
            # Get pre-allocated key and increment index
            child_key = all_keys[key_idx % max_nodes]
            key_idx += 1
                
            # Generate children with TPU optimization
            children = self._generate_thoughts(
                thought.embeddings, 
                thought, 
                depth + 1, 
                thought.content, 
                child_key
            )
            
            # Add children to queue
            for i, child in enumerate(children):
                queue.append((child, depth + 1, key_idx + i))
                key_idx += 1
        
        return best_thought
    
    def _monte_carlo_tree_search(self, initial_thoughts: List[Thought], rng_key: jnp.ndarray) -> Thought:
        """
        Perform Monte Carlo Tree Search with TPU optimization.
        
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
        
        # Optimization: pre-compute sqrt(log(N)) values for common N
        log_cache = {n: np.sqrt(np.log(n)) for n in range(1, 1000)}
        
        def get_uct_score(thought):
            """TPU-optimized UCT scoring."""
            exploit = scores[thought.content]
            total_visits = sum(visits.values()) + 1
            visit_count = visits[thought.content] + 1e-6
            
            # Use cache for sqrt(log(n)) when possible
            if total_visits < 1000:
                explore = self.config.exploration_factor * log_cache[total_visits] / np.sqrt(visit_count)
            else:
                explore = self.config.exploration_factor * np.sqrt(np.log(total_visits) / visit_count)
                
            return exploit + explore
        
        # Generate all random keys at once for better TPU efficiency
        num_iterations = self.config.max_thoughts * self.config.max_depth
        all_keys = jax.random.split(rng_key, num_iterations)
        
        # Run MCTS iterations
        for i in range(num_iterations):
            # Selection: use UCT to pick node
            current = max(initial_thoughts, key=lambda t: get_uct_score(t))
            depth = 0
            
            # Pre-allocated key for expansion
            current_key = all_keys[i]
            
            # Simulation/expansion
            while depth < self.config.max_depth:
                # Generate children with TPU optimization
                children = self._generate_thoughts(
                    current.embeddings, 
                    current, 
                    depth + 1, 
                    current.content, 
                    current_key
                )
                
                if not children:
                    break
                    
                # Select best child
                current = max(children, key=lambda t: t.score)
                visits[current.content] = visits.get(current.content, 0) + 1
                scores[current.content] = scores.get(current.content, current.score)
                
                # Update best thought
                if current.score > best_thought.score:
                    best_thought = current
                    
                depth += 1
        
        return best_thought
