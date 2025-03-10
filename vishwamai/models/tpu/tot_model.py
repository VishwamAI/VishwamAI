"""
TPU-optimized Tree of Thoughts (ToT) model using JAX/XLA
"""

import jax
import jax.numpy as jnp
import haiku as hk
from typing import Optional, List, Tuple, Dict, Any
import math
from collections import deque

# Configure JAX for TPU
jax.config.update("jax_enable_x64", False)  # Disable float64 for TPU efficiency
jax.config.update("jax_default_matmul_precision", "bfloat16")  # Use bfloat16 for matrix operations
jax.config.update("jax_platforms", "tpu")  # Force TPU platform
jax.config.update("jax_xla_backend", "tpu")  # Use TPU XLA backend

from .cot_model import CoTModelTPU
from .kernel_layers import TPUGEMMLinear, TPULayerNorm

def generate_tot(model: 'ToTModelTPU', input_ids: jnp.ndarray, max_length: int = 512,
                temperature: float = 0.6, top_p: float = 0.95,
                search_method: str = "bfs", beam_size: int = 5) -> jnp.ndarray:
    """Standalone generation function for ToT outputs with tree search.
    
    Args:
        model: The ToT model instance
        input_ids: Input token IDs
        max_length: Maximum sequence length
        temperature: Sampling temperature
        top_p: Nucleus sampling probability threshold
        search_method: Search strategy ("bfs" or "dfs")
        beam_size: Number of branches to explore at each step
        
    Returns:
        Generated token IDs incorporating reasoning steps from tree search
    """
    return model.generate_tot(input_ids, max_length, temperature, top_p, search_method, beam_size)

class ThoughtNodeTPU:
    """Represents a node in the thought tree with TPU state management"""
    def __init__(self, thought_text: str, node_id: str, parent=None,
                 score: float = 0.0, hidden_state: Optional[jnp.ndarray] = None):
        self.thought_text = thought_text
        self.node_id = node_id
        self.parent = parent
        self.children = []
        self.score = score
        self.depth = parent.depth + 1 if parent else 0
        self.hidden_state = hidden_state

    def add_child(self, child: 'ThoughtNodeTPU') -> None:
        self.children.append(child)

    def get_state_dict(self) -> Dict[str, any]:
        """Get node state for persistence"""
        return {
            'text': self.thought_text,
            'node_id': self.node_id,
            'parent_id': self.parent.node_id if self.parent else None,
            'score': float(self.score),  # Convert from DeviceArray to float
            'depth': self.depth,
            'child_ids': [child.node_id for child in self.children]
        }

    def path_to_root(self) -> List[str]:
        """Return path from this node to root"""
        path = []
        current = self
        while current:
            path.append(current.thought_text)
            current = current.parent
        return list(reversed(path))

class ToTModelTPU(CoTModelTPU):
    """Tree of Thoughts model with TPU optimization"""
    
    def __init__(self, embed_dim=512, num_layers=12, num_heads=8, ff_dim=2048,
                 vocab_size=50000, max_seq_len=512, num_experts=7, 
                 max_thoughts=5, max_depth=10, dropout_rate=0.1,
                 name: Optional[str] = None):
        super().__init__(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_experts=num_experts,
            dropout_rate=dropout_rate,
            name=name
        )
        self.max_thoughts = max_thoughts
        self.max_depth = max_depth
        self.eval_head = TPUGEMMLinear(3)  # 3 classes for evaluation
        self.thought_generator = TPUGEMMLinear(vocab_size)

    @jax.jit
    def evaluate_thought(self, thought_ids: jnp.ndarray) -> jnp.ndarray:
        """Evaluate a thought's likelihood of leading to a solution"""
        logits, _ = self(thought_ids, is_training=False)
        last_hidden = logits[:, -1, :]
        eval_logits = self.eval_head(last_hidden)
        probs = jax.nn.softmax(eval_logits, axis=-1)
        return probs[:, 0]  # Return "sure" probability

    def generate_candidates(self, input_text: str, current_thought: str,
                          tokenizer, num_candidates: int,
                          tree_id: Optional[str] = None) -> List[str]:
        """Generate candidate thoughts using JAX random sampling"""
        prompt = f"{input_text}\nCurrent thought: {current_thought}\nPropose next steps:"
        input_ids = tokenizer.encode(prompt, return_tensors="jax")

        candidates = []
        for i in range(num_candidates):
            rng = hk.next_rng_key()
            output_ids = self._sample(input_ids, rng, max_length=50)
            candidate_text = tokenizer.decode(output_ids[0]).replace(prompt, "").strip()
            if candidate_text and candidate_text not in candidates:
                candidates.append(candidate_text)

        return candidates[:num_candidates]

    def solve_with_tot(self, input_text: str, tokenizer, search_method: str = "bfs",
                      b: int = 5, tree_id: Optional[str] = None) -> str:
        """Solve using Tree of Thoughts with BFS or DFS"""
        if tree_id is None:
            tree_id = f"tree_{hash(input_text)}"

        root = ThoughtNodeTPU("Start", node_id=f"{tree_id}_root", score=1.0)
        search_fn = self._bfs_search if search_method.lower() == "bfs" else self._dfs_search
        
        final_node = search_fn(input_text, root, tokenizer, b, tree_id)

        if final_node:
            thought_path = final_node.path_to_root()[1:]
            thought_text = " -> ".join(thought_path)
            answer = thought_path[-1].split("=")[-1].strip() if "=" in thought_path[-1] else "No solution found"
            return f"<think>{thought_text}</think> <answer>{answer}</answer>"
        return "<think>Failed to find a solution.</think> <answer>No solution</answer>"

    def _sample(self, input_ids: jnp.ndarray, rng: jnp.ndarray,
                max_length: int = 50, temperature: float = 0.8,
                top_p: float = 0.9) -> jnp.ndarray:
        """TPU-optimized sampling with JAX"""
        def sample_step(carry, _):
            ids, rng = carry
            rng, new_rng = jax.random.split(rng)
            logits, _ = self(ids, is_training=False)
            next_token_logits = logits[:, -1, :] / temperature
            
            # Nucleus sampling
            sorted_logits, sorted_indices = jax.lax.top_k(
                next_token_logits, k=self.vocab_size
            )
            cumulative_probs = jnp.cumsum(jax.nn.softmax(sorted_logits), axis=-1)
            mask = cumulative_probs < top_p
            mask = jnp.concatenate([
                jnp.ones_like(mask[:, :1]),
                mask[:, :-1]
            ], axis=-1)
            
            next_token_logits = jnp.where(
                mask,
                sorted_logits,
                jnp.full_like(sorted_logits, -float('inf'))
            )
            next_token = jax.random.categorical(rng, next_token_logits)
            next_token = sorted_indices[jnp.arange(next_token.shape[0]), next_token]
            
            return (jnp.concatenate([ids, next_token[:, None]], axis=1), new_rng), None

        final_state, _ = jax.lax.scan(
            sample_step,
            (input_ids, rng),
            None,
            length=max_length - input_ids.shape[1]
        )
        return final_state[0]

    def _bfs_search(self, input_text: str, root: ThoughtNodeTPU, tokenizer,
                    b: int, tree_id: str) -> Optional[ThoughtNodeTPU]:
        """Perform BFS with TPU-optimized batch processing"""
        queue = deque([root])
        step = 0
        best_node = None
        best_score = -float('inf')

        while queue and step < self.max_steps:
            level_size = len(queue)
            level_nodes = []

            # Process level in batches for TPU efficiency
            batch_size = 8  # Adjust based on TPU memory
            for batch_start in range(0, level_size, batch_size):
                batch_nodes = list(queue)[:batch_size]
                queue.rotate(-batch_size)

                # Generate candidates for batch
                all_candidates = []
                for node in batch_nodes:
                    candidates = self.generate_candidates(
                        input_text, node.thought_text,
                        tokenizer, self.candidates_per_step, tree_id
                    )
                    all_candidates.extend([
                        (cand, node) for cand in candidates
                    ])

                # Evaluate candidates in parallel
                if all_candidates:
                    texts = [c[0] for c in all_candidates]
                    input_ids = tokenizer.batch_encode(texts, return_tensors="jax")
                    scores = self.evaluate_thought(input_ids)

                    # Create nodes for all candidates
                    for i, ((cand_text, parent), score) in enumerate(zip(all_candidates, scores)):
                        node_id = f"{tree_id}_level{step}_cand{len(level_nodes) + i}"
                        score = float(score)  # Convert from DeviceArray
                        child_node = ThoughtNodeTPU(
                            thought_text=cand_text,
                            node_id=node_id,
                            parent=parent,
                            score=score
                        )
                        parent.add_child(child_node)
                        level_nodes.append(child_node)

                        if score > best_score and "24" in cand_text and "=" in cand_text:
                            best_node = child_node
                            best_score = score

            # Sort and filter level nodes
            level_nodes.sort(key=lambda x: x.score, reverse=True)
            queue.extend(level_nodes[:b])
            step += 1

        return best_node

    def _dfs_search(self, input_text: str, root: ThoughtNodeTPU, tokenizer,
                    b: int, tree_id: str) -> Optional[ThoughtNodeTPU]:
        """Perform DFS with TPU-optimized batch processing"""
        stack = [(root, 0)]
        best_node = None
        best_score = -float('inf')
        visited = set()

        while stack:
            current_node, step = stack.pop()
            if step >= self.max_steps or current_node.depth >= self.max_depth:
                continue

            # Generate and evaluate candidates
            candidates = self.generate_candidates(
                input_text, current_node.thought_text,
                tokenizer, self.candidates_per_step, tree_id
            )

            if candidates:
                # Batch evaluate candidates
                input_ids = tokenizer.batch_encode(candidates, return_tensors="jax")
                scores = self.evaluate_thought(input_ids)

                # Create nodes for all candidates
                child_nodes = []
                for i, (cand_text, score) in enumerate(zip(candidates, scores)):
                    node_id = f"{tree_id}_step{step}_cand{i}"
                    score = float(score)  # Convert from DeviceArray
                    child_node = ThoughtNodeTPU(
                        thought_text=cand_text,
                        node_id=node_id,
                        parent=current_node,
                        score=score
                    )
                    current_node.add_child(child_node)
                    child_nodes.append(child_node)

                    if score > best_score and "24" in cand_text and "=" in cand_text:
                        best_node = child_node
                        best_score = score

                # Sort and add best candidates to stack
                child_nodes.sort(key=lambda x: x.score, reverse=True)
                stack.extend((node, step + 1) for node in child_nodes[:b])

        return best_node

    def generate_tot(self, input_ids: jnp.ndarray, max_length: int = 512,
                    temperature: float = 0.6, top_p: float = 0.95,
                    search_method: str = "bfs", beam_size: int = 5) -> jnp.ndarray:
        """Generate output using Tree of Thoughts with specified search strategy.
        
        Args:
            input_ids: Input token IDs
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_p: Nucleus sampling probability threshold
            search_method: Search strategy ("bfs" or "dfs")
            beam_size: Number of branches to explore at each step
            
        Returns:
            Generated token IDs incorporating reasoning steps
        """
        # Create root thought
        root = ThoughtNodeTPU("Start", node_id="root", score=1.0)
        
        # Use appropriate search strategy
        search_fn = self._bfs_search if search_method.lower() == "bfs" else self._dfs_search
        final_node = search_fn(input_ids, root, beam_size)
        
        if final_node:
            # Convert thought path to token sequence
            thought_path = final_node.path_to_root()[1:]  # Skip root node
            
            @jax.jit
            def generate_sequence():
                # Initialize with input
                sequence = input_ids
                
                # Add each thought step
                for thought in thought_path:
                    # Generate tokens for this thought
                    thought_ids = self.generate_cot(
                        sequence,
                        max_length=max_length,
                        temperature=temperature,
                        top_p=top_p
                    )
                    sequence = thought_ids
                
                return sequence
                
            return generate_sequence()
        
        # Fallback to standard generation if search fails
        return self.generate_cot(input_ids, max_length, temperature, top_p)

# Example usage and test
if __name__ == "__main__":
    def run_tot(x: jnp.ndarray) -> jnp.ndarray:
        model = ToTModelTPU()
        return model(x, is_training=False)[0]

    # Initialize
    batch_size, seq_len = 2, 64
    rng = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 50000)

    # Transform and initialize
    transformed = hk.transform(run_tot)
    params = transformed.init(rng, input_ids)

    # Forward pass
    logits = transformed.apply(params, rng, input_ids)
    print("ToT Output shape:", logits.shape)