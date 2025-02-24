import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from .transformer import VisionTransformer10B
import logging

logger = logging.getLogger(__name__)

@dataclass
class Thought:
    content: str
    score: float
    children: List['Thought']
    parent: Optional['Thought'] = None

class ThoughtGenerator(nn.Module):
    hidden_size: int
    vocab_size: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Generates a thought representation from an input tensor.
        
        Transforms the input tensor through a dense layer with a ReLU activation followed
        by a second dense layer to produce output logits corresponding to the vocabulary.
        
        Args:
            x (jnp.ndarray): Input tensor to be transformed.
        
        Returns:
            jnp.ndarray: Output tensor containing thought logits.
        """
        logger.info("Generating thoughts")
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.vocab_size)(x)
        return x

class ThoughtEvaluator(nn.Module):
    hidden_size: int
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        """
        Evaluates an input tensor to produce thought evaluation probabilities.
        
        This method applies a dense hidden layer with a ReLU activation, followed by a dense layer
        that outputs scores for three categories: 'sure', 'maybe', and 'impossible'. The final
        output is computed using a softmax to yield a probability distribution over these outcomes.
        
        Args:
            x (jnp.ndarray): Input feature tensor.
        
        Returns:
            jnp.ndarray: Probability distribution over the three evaluation categories.
        """
        logger.info("Evaluating thoughts")
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)  # 3 scores: sure, maybe, impossible
        return nn.softmax(x)

class TreeOfThoughts(nn.Module):
    transformer: VisionTransformer10B
    max_thoughts: int = 5
    max_depth: int = 3
    
    def setup(self) -> None:
        """
        Initializes the neural network components for thought generation and evaluation.
        
        This method instantiates the ThoughtGenerator with a hidden size of 1024 and a vocabulary
        size of 32000, along with the ThoughtEvaluator using a hidden size of 512. These components
        are used to generate thought representations and assess their quality within the Tree of
        Thoughts framework.
        """
        self.thought_generator = ThoughtGenerator(hidden_size=1024, vocab_size=32000)
        self.thought_evaluator = ThoughtEvaluator(hidden_size=512)
        
    def generate_thoughts(self, x: jnp.ndarray, k: int = 5) -> List[Thought]:
        """
        Generate thoughts using the transformer and thought generator.

        Args:
            x (jnp.ndarray): Input tensor.
            k (int): Number of thoughts to generate.

        Returns:
            List[Thought]: List of generated thoughts.
        """
        logger.info("Generating thoughts")
        features = self.transformer(x)
        logits = self.thought_generator(features)
        # Sample k thoughts using temperature sampling
        thoughts = jax.random.categorical(jax.random.PRNGKey(0), logits, shape=(k,))
        return [Thought(content=t, score=0.0, children=[]) for t in thoughts]
    
    def evaluate_thought(self, thought: Thought) -> jnp.ndarray:
        """
        Evaluates a thought by converting its text into embeddings and scoring it.
        
        This method transforms the thought's content into a numeric array where each
        element is the ordinal value of the corresponding character, and then passes
        this array to the neural evaluator to compute scores.
        
        Args:
            thought: A Thought instance containing the text to be evaluated.
        
        Returns:
            jnp.ndarray: An array of scores computed by the thought evaluator.
        """
        logger.info("Evaluating thought")
        # Convert thought to embeddings (simplified)
        x = jnp.array([ord(c) for c in thought.content])
        scores = self.thought_evaluator(x)
        return scores
    
    def bfs_search(self, initial_state: jnp.ndarray, max_width: int = 5) -> Thought:
        """
        Performs a breadth-first search on a thought tree to locate the best thought.
        
        This method begins with an initial state and iteratively generates thoughts using the
        search's branching factor. Each thought is evaluated using its primary "sure" score,
        and if a thought's children have not reached the maximum depth, new children are generated.
        The search continues until all generated thoughts are examined, and the thought with the
        highest score is returned.
        
        Args:
            initial_state (jnp.ndarray): The input tensor from which to generate initial thoughts.
            max_width (int): The maximum number of thoughts to generate at each node expansion.
        
        Returns:
            Thought: The thought with the highest evaluation score found during the search.
        """
        logger.info("Starting BFS search")
        queue = self.generate_thoughts(initial_state, k=max_width)
        best_thought = None
        best_score = float('-inf')
        
        while queue:
            current = queue.pop(0)
            scores = self.evaluate_thought(current)
            current.score = float(scores[0])  # "sure" score
            
            if current.score > best_score:
                best_thought = current
                best_score = current.score
                
            if len(current.children) < self.max_depth:
                children = self.generate_thoughts(current.content, k=max_width)
                for child in children:
                    child.parent = current
                    current.children.append(child)
                queue.extend(children)
                
        return best_thought
    
    def dfs_search(self, initial_state: jnp.ndarray, max_width: int = 5) -> Thought:
        """
        Performs a depth-first search to identify the thought with the highest score.
        
        This method begins by generating initial thoughts from the provided state tensor and
        iteratively explores the thought tree in a depth-first manner. Each thought is evaluated
        using a scoring mechanism, and child thoughts are generated for nodes that have not reached
        the maximum search depth. The thought with the highest observed score is returned.
        
        Args:
            initial_state (jnp.ndarray): The tensor representing the starting state for thought generation.
            max_width (int): The maximum number of thoughts to generate at each branch expansion.
        
        Returns:
            Thought: The thought instance with the best score found during the search.
        """
        logger.info("Starting DFS search")
        stack = self.generate_thoughts(initial_state, k=max_width)
        best_thought = None
        best_score = float('-inf')
        
        while stack:
            current = stack.pop()
            scores = self.evaluate_thought(current)
            current.score = float(scores[0])
            
            if current.score > best_score:
                best_thought = current
                best_score = current.score
                
            if len(current.children) < self.max_depth:
                children = self.generate_thoughts(current.content, k=max_width)
                for child in children:
                    child.parent = current
                    current.children.append(child)
                stack.extend(children)
                
        return best_thought
    
    def __call__(self, x: jnp.ndarray, search_strategy: str = 'bfs') -> Thought:
        """
        Perform a search strategy to select the best thought.
        
        Executes a breadth-first (BFS) or depth-first (DFS) search starting from the given
        input tensor to identify the optimal thought. The search strategy determines
        the traversal method over the thought space.
        
        Args:
            x (jnp.ndarray): Input state tensor.
            search_strategy (str): The search method to use, either 'bfs' for breadth-first
                search or 'dfs' for depth-first search.
        
        Returns:
            Thought: The best thought discovered by the search.
        
        Raises:
            ValueError: If an unsupported search strategy is specified.
        """
        logger.info(f"Performing {search_strategy} search")
        if search_strategy == 'bfs':
            return self.bfs_search(x)
        elif search_strategy == 'dfs':
            return self.dfs_search(x)
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")

def create_tot_optimizer(learning_rate: float = 1e-4) -> optax.GradientTransformation:
    """
    Creates an Adam optimizer for training Tree of Thoughts models.
    
    Args:
        learning_rate (float): Learning rate to use for the Adam optimizer.
    
    Returns:
        optax.GradientTransformation: The configured Adam optimizer.
    """
    logger.info("Creating ToT optimizer")
    return optax.adam(learning_rate)
