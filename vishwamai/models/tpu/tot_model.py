"""
Tree of Thoughts (ToT) model for VishwamAI, extending the CoT model with tree search.
Supports DFS and BFS for thought exploration, thought generation, and evaluation.
Designed for complex reasoning tasks requiring deep calculations.
Updated for JAX/Flax/DM-Haiku, optimized for TPUs.
"""

import jax
import jax.numpy as jnp
from jax import random, jit, lax
import flax.linen as nn
import haiku as hk
from typing import Optional, Tuple, List, Dict
import numpy as np

from vishwamai.models.transformer import VishwamAITransformer
from vishwamai.models.cot_model import CoTModel, extract_answer

# Define ThoughtNode as a Haiku-transformable structure
def thought_node_fn(thought_text: str, parent=None, score: float = 0.0):
    """Haiku-compatible function to create a ThoughtNode."""
    state = hk.get_state("thought_state", shape=(), init=lambda *args: 0)
    depth = state + (parent.depth if parent else -1) + 1 if parent else 0
    hk.set_state("thought_state", depth)
    return {
        "thought_text": thought_text,
        "parent": parent,
        "children": [],
        "score": score,
        "depth": depth
    }

class ThoughtNode:
    """Wrapper for Haiku-transformed ThoughtNode state."""
    def __init__(self, thought_text: str, parent=None, score: float = 0.0):
        self._state = hk.transform(lambda: thought_node_fn(thought_text, parent, score))
        self._rng = None

    def init(self, rng: jnp.ndarray):
        """Initialize the Haiku state with a random key."""
        self._rng, sub_rng = jax.random.split(rng)
        params = self._state.init(sub_rng)
        return params

    def apply(self, params, rng: jnp.ndarray):
        """Apply the Haiku function with the given parameters and RNG."""
        self._rng, sub_rng = jax.random.split(rng)
        return self._state.apply(params, sub_rng)

    def add_child(self, child):
        """Add a child node (simplified for JAX compatibility)."""
        self.children.append(child)

    @property
    def children(self):
        return self._state.state["children"]

    @children.setter
    def children(self, value):
        self._state.state["children"] = value

    @property
    def thought_text(self):
        return self._state.state["thought_text"]

    @property
    def parent(self):
        return self._state.state["parent"]

    @property
    def score(self):
        return self._state.state["score"]

    @property
    def depth(self):
        return self._state.state["depth"]

    def path_to_root(self):
        """Return the path from this node to the root as a list of thoughts."""
        path = []
        current = self
        while current:
            path.append(current.thought_text)
            current = current.parent
        return list(reversed(path))

class ToTModel(CoTModel):
    """
    Tree of Thoughts model extending CoTModel with tree search capabilities.
    """
    max_steps: int = 3
    candidates_per_step: int = 5
    max_depth: int = 10

    def setup(self):
        """
        Initialize the ToT model with a VishwamAI Transformer and evaluation head.
        """
        super().setup()
        self.eval_head = nn.Dense(3)  # Outputs logits for 3 classes (sure/maybe/impossible)

    @nn.compact
    def evaluate_thought(self, thought_ids: jnp.ndarray, train: bool = False) -> float:
        """
        Evaluate a thought by predicting its likelihood of leading to a solution.
        
        Args:
            thought_ids: Token IDs of the thought (batch_size, seq_len).
            train: Whether in training mode.
        
        Returns:
            Float: Probability of "sure".
        """
        logits = self.transformer(thought_ids, train=train)
        last_hidden = self.transformer.get_hidden_state(thought_ids, train=train)[:, -1, :]
        eval_logits = self.eval_head(last_hidden)  # (batch_size, 3)
        probs = jax.nn.softmax(eval_logits, axis=-1)
        return probs[:, 0].mean()  # Average "sure" probability across batch

    def generate_candidates(self, input_text: str, current_thought: str, tokenizer, num_candidates: int, 
                           rng: jnp.ndarray) -> List[str]:
        """
        Generate candidate thoughts for the next step.
        
        Args:
            input_text: Original input problem.
            current_thought: Current thought text.
            tokenizer: Tokenizer instance.
            num_candidates: Number of candidates to generate.
            rng: Random number generator key.
        
        Returns:
            List of candidate thought texts.
        """
        prompt = f"{input_text}\nCurrent thought: {current_thought}\nPropose next steps:"
        input_ids = jnp.array(tokenizer.encode(prompt, return_tensors="jax")[0], dtype=jnp.int32)

        def body_fn(val):
            i, candidates, rng = val
            if i >= num_candidates:
                return i, candidates, rng
            output_ids = self._sample(input_ids[None, :], 50, 0.8, 0.9, rng)
            candidate_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            candidate_text = candidate_text.replace(prompt, "").strip()
            if candidate_text:
                candidates = candidates.at[i].set(candidate_text)
            rng, sub_rng = jax.random.split(rng)
            return i + 1, candidates, sub_rng

        candidates = jnp.array([""] * num_candidates)
        _, candidates, _ = lax.while_loop(
            lambda val: val[0] < num_candidates,
            body_fn,
            (0, candidates, rng)
        )
        return [cand for cand in candidates if cand]

    @jit
    def _bfs_search(self, input_text: str, root: ThoughtNode, tokenizer, b: int, rng: jnp.ndarray) -> Optional[ThoughtNode]:
        """
        Perform BFS to explore the thought tree.
        
        Args:
            input_text: Input problem.
            root: Root node of the thought tree.
            tokenizer: Tokenizer instance.
            b: Number of best candidates to keep at each step.
            rng: Random number generator key.
        
        Returns:
            ThoughtNode: Node with the final solution, or None.
        """
        queue = [root]
        step = 0

        def body_fn(val):
            nonlocal step
            queue, step = val
            if not queue or step >= self.max_steps:
                return queue, step
            current_nodes = queue
            new_queue = []
            for node in current_nodes:
                if node.depth >= self.max_depth:
                    continue
                candidates = self.generate_candidates(input_text, node.thought_text, tokenizer, self.candidates_per_step, rng)
                for candidate in candidates:
                    candidate_ids = jnp.array(tokenizer.encode(candidate, return_tensors="jax")[0], dtype=jnp.int32)
                    score = self.evaluate_thought(candidate_ids[None, :])
                    child = ThoughtNode(candidate, node, score)
                    node.add_child(child)
                    new_queue.append(child)
            # Keep top b candidates
            if new_queue:
                scored_nodes = [(n, n.score) for n in new_queue]
                scored_nodes.sort(key=lambda x: x[1], reverse=True)
                new_queue = [n for n, _ in scored_nodes[:b]]
            step += 1
            return new_queue, step

        queue, _ = lax.while_loop(
            lambda val: val[0] and val[1] < self.max_steps,
            body_fn,
            (queue, step)
        )

        if queue:
            best_node = max(queue, key=lambda x: x.score)
            if "24" in best_node.thought_text and "=" in best_node.thought_text:
                return best_node
        return None

    @jit
    def _dfs_search(self, input_text: str, root: ThoughtNode, tokenizer, b: int, rng: jnp.ndarray) -> Optional[ThoughtNode]:
        """
        Perform DFS to explore the thought tree.
        
        Args:
            input_text: Input problem.
            root: Root node of the thought tree.
            tokenizer: Tokenizer instance.
            b: Number of best candidates to keep at each step.
            rng: Random number generator key.
        
        Returns:
            ThoughtNode: Node with the final solution, or None.
        """
        stack = [(root, 0)]
        best_node = None
        best_score = -jnp.inf

        def body_fn(val):
            nonlocal best_node, best_score
            stack = val
            if not stack:
                return stack
            current_node, step = stack[-1]
            stack = stack[:-1]
            if step >= self.max_steps or current_node.depth >= self.max_depth:
                return stack

            candidates = self.generate_candidates(input_text, current_node.thought_text, tokenizer, self.candidates_per_step, rng)
            scored_candidates = []
            for candidate in candidates:
                candidate_ids = jnp.array(tokenizer.encode(candidate, return_tensors="jax")[0], dtype=jnp.int32)
                score = self.evaluate_thought(candidate_ids[None, :])
                child = ThoughtNode(candidate, current_node, score)
                current_node.add_child(child)
                scored_candidates.append(child)
                if "24" in child.thought_text and "=" in child.thought_text and score > best_score:
                    best_node = child
                    best_score = score

            # Keep top b candidates
            scored_candidates.sort(key=lambda x: x.score, reverse=True)
            top_candidates = scored_candidates[:b]
            stack.extend((c, step + 1) for c in top_candidates)
            return stack

        stack = lax.while_loop(
            lambda val: val,
            body_fn,
            stack
        )

        return best_node

    def solve_with_tot(self, input_text: str, tokenizer, search_method: str = "bfs", b: int = 5, 
                       rng: jnp.ndarray = None) -> str:
        """
        Solve the problem using Tree of Thoughts with BFS or DFS.
        
        Args:
            input_text: Input problem (e.g., "Game of 24: 4 9 10 13").
            tokenizer: Tokenizer instance.
            search_method: "bfs" or "dfs" for search strategy.
            b: Number of best candidates to keep at each step.
            rng: Random number generator key.
        
        Returns:
            Final output with <think> and <answer> tags.
        """
        if rng is None:
            rng = random.PRNGKey(0)
        
        # Initialize the thought tree
        root = ThoughtNode(thought_text="Start", score=1.0)
        root_params = root.init(rng)

        if search_method.lower() == "bfs":
            final_node = self._bfs_search(input_text, root, tokenizer, b, rng)
        else:
            final_node = self._dfs_search(input_text, root, tokenizer, b, rng)

        # Construct the final output
        if final_node:
            thought_path = final_node.path_to_root()[1:]  # Exclude "Start"
            thought_text = " -> ".join(thought_path)
            answer = thought_path[-1].split("=")[-1].strip() if "=" in thought_path[-1] else "No solution found"
            return f"<think>{thought_text}</think> <answer>{answer}</answer>"
        else:
            return "<think>Failed to find a solution.</think> <answer>No solution</answer>"

# Example usage
if __name__ == "__main__":
    # Mock tokenizer (same as in cot_model.py)
    class MockTokenizer:
        def __init__(self, vocab_size=50000):
            self.vocab_size = vocab_size
            self.special_tokens = {
                "<think>": vocab_size-4, "</think>": vocab_size-3,
                "<answer>": vocab_size-2, "</answer>": vocab_size-1
            }
            self.inverse_vocab = {v: k for k, v in self.special_tokens.items()}
            self.inverse_vocab.update({i: f"token_{i}" for i in range(vocab_size-4)})

        def encode(self, text, return_tensors="jax"):
            tokens = [self.special_tokens.get(text, i) for i in range(5)]  # Simplified
            if return_tensors == "jax":
                return jnp.array([tokens], dtype=jnp.int32)
            return tokens

        def decode(self, token_ids, skip_special_tokens=False):
            text = []
            for token in token_ids:
                token = int(token)
                if token in self.inverse_vocab:
                    if not skip_special_tokens or token < self.vocab_size-4:
                        text.append(self.inverse_vocab[token])
            return " ".join(text)

    # Initialize model and tokenizer
    rng = random.PRNGKey(0)
    rng, init_rng = random.split(rng)
    tokenizer = MockTokenizer()
    model = ToTModel(vocab_size=tokenizer.vocab_size)
    params = model.init(init_rng, jnp.ones((1, 5), dtype=jnp.int32))['params']

    # Example: Solve a Game of 24 problem
    input_text = "Game of 24: 4 9 10 13"
    output_bfs = model.solve_with_tot(input_text, tokenizer, search_method="bfs", b=5, rng=rng)
    print("BFS Output:", output_bfs)
    print("BFS Extracted Answer:", extract_answer(output_bfs))

    rng, dfs_rng = random.split(rng)
    output_dfs = model.solve_with_tot(input_text, tokenizer, search_method="dfs", b=5, rng=dfs_rng)
    print("DFS Output:", output_dfs)
    print("DFS Extracted Answer:", extract_answer(output_dfs))