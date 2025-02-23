import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
from .transformer import VisionTransformer10B

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
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.vocab_size)(x)
        return x

class ThoughtEvaluator(nn.Module):
    hidden_size: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(3)(x)  # 3 scores: sure, maybe, impossible
        return nn.softmax(x)

class TreeOfThoughts(nn.Module):
    transformer: VisionTransformer10B
    max_thoughts: int = 5
    max_depth: int = 3
    
    def setup(self):
        self.thought_generator = ThoughtGenerator(hidden_size=1024, vocab_size=32000)
        self.thought_evaluator = ThoughtEvaluator(hidden_size=512)
        
    def generate_thoughts(self, x, k: int = 5) -> List[Thought]:
        features = self.transformer(x)
        logits = self.thought_generator(features)
        # Sample k thoughts using temperature sampling
        thoughts = jax.random.categorical(jax.random.PRNGKey(0), logits, shape=(k,))
        return [Thought(content=t, score=0.0, children=[]) for t in thoughts]
    
    def evaluate_thought(self, thought: Thought) -> jnp.ndarray:
        # Convert thought to embeddings (simplified)
        x = jnp.array([ord(c) for c in thought.content])
        scores = self.thought_evaluator(x)
        return scores
    
    def bfs_search(self, initial_state, max_width: int = 5) -> Thought:
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
    
    def dfs_search(self, initial_state, max_width: int = 5) -> Thought:
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
    
    def __call__(self, x, search_strategy: str = 'bfs'):
        if search_strategy == 'bfs':
            return self.bfs_search(x)
        elif search_strategy == 'dfs':
            return self.dfs_search(x)
        else:
            raise ValueError(f"Unknown search strategy: {search_strategy}")

def create_tot_optimizer(learning_rate: float = 1e-4):
    return optax.adam(learning_rate)
