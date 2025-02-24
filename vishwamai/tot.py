import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import freeze, unfreeze
import optax
from typing import List, Tuple, Dict, Optional, NamedTuple, Any
from dataclasses import dataclass
from .transformer import VisionTransformer10B
import logging
from functools import partial

logger = logging.getLogger(__name__)

class ThoughtTokenizer:
    """Tokenizer for converting between thoughts and token IDs."""
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        # In practice, you would load a real vocabulary and embeddings
        self.embedding_dim = 768
        self.embeddings = jax.random.normal(
            jax.random.PRNGKey(0), 
            (vocab_size, self.embedding_dim)
        )
    
    def encode(self, text: str) -> jnp.ndarray:
        """Convert text to token IDs."""
        # Simplified tokenization (in practice, use a proper tokenizer)
        return jnp.array([hash(word) % self.vocab_size for word in text.split()])
    
    def decode(self, token_ids: jnp.ndarray) -> str:
        """Convert token IDs back to text."""
        # Simplified decoding (in practice, use a proper tokenizer)
        return " ".join([str(int(id)) for id in token_ids])
    
    def get_embeddings(self, token_ids: jnp.ndarray) -> jnp.ndarray:
        """Get embeddings for token IDs."""
        return self.embeddings[token_ids]

@dataclass
class Thought:
    content: str  # Raw text content
    token_ids: jnp.ndarray  # Token IDs
    embeddings: jnp.ndarray  # Token embeddings
    score: float
    children: List['Thought']
    parent: Optional['Thought'] = None

class SearchState(NamedTuple):
    """Enhanced state tracking for tree search."""
    best_score: float
    best_thought: Optional[Thought]
    num_thoughts: int
    depth: int
    value_cache: Dict[str, float]  # Cache thought evaluations
    beam_width: int
    exploration_factor: float  # Controls exploration vs exploitation
    pruning_threshold: float  # Threshold for pruning low-value branches

class ThoughtGenerator(nn.Module):
    """Generate thoughts from input features."""
    hidden_size: int
    vocab_size: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray, key: jax.random.PRNGKey) -> jnp.ndarray:
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        logits = nn.Dense(self.vocab_size)(x)
        
        # Temperature sampling for diverse thought generation
        temperature = 0.7
        logits = logits / temperature
        return jax.random.categorical(key, logits)

class ThoughtEvaluator(nn.Module):
    """Evaluate thoughts using learned embeddings."""
    hidden_size: int
    
    @nn.compact
    def __call__(self, embeddings: jnp.ndarray) -> jnp.ndarray:
        # Project embeddings through MLP
        x = nn.Dense(self.hidden_size)(embeddings)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size // 2)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)  # 3 scores: sure, maybe, impossible
        return nn.softmax(x)

def scan_expand_thoughts(carry: SearchState, _: Any, 
                        params: Dict, 
                        generator: ThoughtGenerator,
                        evaluator: ThoughtEvaluator,
                        tokenizer: ThoughtTokenizer,
                        key: jax.random.PRNGKey,
                        max_width: int) -> Tuple[SearchState, List[Thought]]:
    """Enhanced thought expansion with beam search and value caching."""
    state, current_thought = carry
    
    if state.num_thoughts >= max_width or state.depth >= 3:
        return carry, []
    
    # Check cache first
    if current_thought.content in state.value_cache:
        cached_score = state.value_cache[current_thought.content]
        if cached_score < state.pruning_threshold:
            return carry, []  # Prune low-value branches
    
    # Generate thoughts in parallel using vmap
    keys = jax.random.split(key, state.beam_width)
    features = current_thought.embeddings.mean(axis=0, keepdims=True)
    
    @partial(jax.vmap, in_axes=(None, 0))
    def generate_thought(feat, subkey):
        token_ids = generator.apply(params['generator'], feat, subkey)
        return token_ids
    
    token_ids_batch = generate_thought(features, keys)
    
    # Evaluate thoughts in parallel
    embeddings_batch = jax.vmap(tokenizer.get_embeddings)(token_ids_batch)
    scores_batch = jax.vmap(lambda x: evaluator.apply(params['evaluator'], x))(embeddings_batch)
    
    # Apply UCB exploration bonus
    exploration_bonus = state.exploration_factor * jnp.sqrt(
        jnp.log(state.num_thoughts + 1) / (jnp.ones(state.beam_width))
    )
    adjusted_scores = scores_batch[:, 0] + exploration_bonus  # Use "sure" scores
    
    # Select top-k thoughts using top_k operation
    top_k = min(state.beam_width, len(adjusted_scores))
    selected_indices = jax.lax.top_k(adjusted_scores, top_k)[1]
    
    new_thoughts = []
    for idx in selected_indices:
        text = tokenizer.decode(token_ids_batch[idx])
        score = float(scores_batch[idx, 0])
        
        # Update value cache
        state.value_cache[text] = score
        
        # Create thought if above pruning threshold
        if score >= state.pruning_threshold:
            new_thought = Thought(
                content=text,
                token_ids=token_ids_batch[idx],
                embeddings=embeddings_batch[idx],
                score=score,
                children=[],
                parent=current_thought
            )
            new_thoughts.append(new_thought)
            
            # Update state
            if score > state.best_score:
                state = state._replace(
                    best_score=score,
                    best_thought=new_thought
                )
    
    # Update state with new statistics
    new_state = state._replace(
        num_thoughts=state.num_thoughts + len(new_thoughts),
        depth=state.depth + 1
    )
    
    return (new_state, current_thought), new_thoughts

class TreeOfThoughts(nn.Module):
    """Enhanced Tree of Thoughts with beam search and adaptive exploration."""
    transformer: VisionTransformer10B
    max_thoughts: int = 5
    max_depth: int = 3
    beam_width: int = 8
    pruning_threshold: float = 0.3
    exploration_factor: float = 1.0
    
    def setup(self):
        self.tokenizer = ThoughtTokenizer()
        self.thought_generator = ThoughtGenerator(hidden_size=1024, vocab_size=32000)
        self.thought_evaluator = ThoughtEvaluator(hidden_size=512)
    
    def generate_thoughts(self, 
                        x: jnp.ndarray, 
                        key: jax.random.PRNGKey,
                        k: int = 5) -> List[Thought]:
        """Generate k thoughts from input features."""
        features = self.transformer(x)
        token_ids = self.thought_generator(features, key)
        
        thoughts = []
        for i in range(k):
            key, subkey = jax.random.split(key)
            text = self.tokenizer.decode(token_ids[i])
            embeddings = self.tokenizer.get_embeddings(token_ids[i])
            thoughts.append(Thought(
                content=text,
                token_ids=token_ids[i],
                embeddings=embeddings,
                score=0.0,
                children=[]
            ))
        return thoughts
    
    def evaluate_thought(self, thought: Thought) -> jnp.ndarray:
        """Evaluate a thought using its embeddings."""
        return self.thought_evaluator(thought.embeddings)
    
    def beam_search(self, 
                   initial_state: jnp.ndarray, 
                   key: jax.random.PRNGKey) -> Thought:
        """Enhanced beam search with adaptive width and pruning."""
        # Initialize beam with parallel thought generation
        initial_thoughts = self.generate_thoughts(initial_state, key, k=self.beam_width)
        
        # Initialize search state with adaptive parameters
        state = SearchState(
            best_score=float('-inf'),
            best_thought=None,
            num_thoughts=len(initial_thoughts),
            depth=0,
            value_cache={},
            beam_width=self.beam_width,
            exploration_factor=self.exploration_factor,
            pruning_threshold=self.pruning_threshold
        )
        
        # Process each level up to max_depth
        for depth in range(self.max_depth):
            # Dynamically adjust beam width based on depth
            adaptive_width = max(
                2, 
                int(self.beam_width * (1.0 - depth / self.max_depth))
            )
            
            # Expand thoughts in parallel for current beam
            expanded_states = []
            expanded_thoughts = []
            
            for thought in initial_thoughts[:adaptive_width]:
                # Generate and evaluate new thoughts
                next_state, next_thoughts = scan_expand_thoughts(
                    (state, thought),
                    None,
                    self.variables,
                    self.thought_generator,
                    self.thought_evaluator,
                    self.tokenizer,
                    key,
                    adaptive_width
                )
                expanded_states.append(next_state)
                expanded_thoughts.extend(next_thoughts)
            
            # Select top-k thoughts for next iteration
            if expanded_thoughts:
                scores = jnp.array([t.score for t in expanded_thoughts])
                top_k_idx = jax.lax.top_k(scores, min(adaptive_width, len(scores)))[1]
                initial_thoughts = [expanded_thoughts[i] for i in top_k_idx]
                
                # Update state with best score from this level
                best_idx = jnp.argmax(scores)
                if expanded_thoughts[best_idx].score > state.best_score:
                    state = state._replace(
                        best_score=expanded_thoughts[best_idx].score,
                        best_thought=expanded_thoughts[best_idx]
                    )
            else:
                break
            
            # Update depth and check termination
            state = state._replace(depth=depth + 1)
            if state.num_thoughts >= self.max_thoughts:
                break
            
            # Generate new random key for next iteration
            key, _ = jax.random.split(key)
        
        return state.best_thought
    
    def dfs_search(self, 
                  initial_state: jnp.ndarray,
                  key: jax.random.PRNGKey) -> Thought:
        """Enhanced depth-first search with pruning and caching."""
        stack = self.generate_thoughts(initial_state, key, k=self.beam_width)
        state = SearchState(
            best_score=float('-inf'),
            best_thought=None,
            num_thoughts=len(stack),
            depth=0,
            value_cache={},
            beam_width=self.beam_width,
            exploration_factor=self.exploration_factor,
            pruning_threshold=self.pruning_threshold
        )
        
        def dfs_step(curr_state: SearchState, curr_thought: Thought, depth: int) -> SearchState:
            if depth >= self.max_depth or curr_state.num_thoughts >= self.max_thoughts:
                return curr_state
            
            # Check cache to avoid redundant exploration
            if curr_thought.content in curr_state.value_cache:
                if curr_state.value_cache[curr_thought.content] < curr_state.pruning_threshold:
                    return curr_state
            
            # Expand current thought
            next_state, new_thoughts = scan_expand_thoughts(
                (curr_state, curr_thought),
                None,
                self.variables,
                self.thought_generator,
                self.thought_evaluator,
                self.tokenizer,
                key,
                self.beam_width
            )
            
            # Recursively explore promising branches
            for thought in new_thoughts:
                if thought.score >= curr_state.pruning_threshold:
                    next_state = dfs_step(next_state, thought, depth + 1)
            
            return next_state
        
        # Process stack in DFS order
        while stack:
            thought = stack.pop()
            state = dfs_step(state, thought, 0)
        
        return state.best_thought
    
    def __call__(self, 
                 x: jnp.ndarray,
                 key: jax.random.PRNGKey,
                 search_strategy: str = 'beam') -> Thought:
        """Perform tree search with the specified strategy."""
        logger.info(f"Performing {search_strategy} search")
        if search_strategy == 'beam':
            return self.beam_search(x, key)
        elif search_strategy == 'dfs':
            return self.dfs_search(x, key)
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}. Supported: 'beam', 'dfs'")

def create_tot_optimizer(learning_rate: float = 1e-4) -> optax.GradientTransformation:
    """Create an optimizer for the Tree of Thoughts."""
    return optax.chain(
        optax.clip_by_global_norm(1.0),  # Gradient clipping
        optax.adam(learning_rate)
    )
