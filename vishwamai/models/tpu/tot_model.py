"""
TPU-optimized Tree of Thoughts (ToT) model using JAX/XLA
"""

import jax
import jax.numpy as jnp
import haiku as hk
import optax
from typing import Optional, List, Tuple, Dict, Any
import math
from collections import deque

from .cot_model import CoTModelTPU
from .kernel_layers import TPUGEMMLinear, TPULayerNorm
from .core import DTYPE_CONFIG

def generate_tot(model: 'ToTModelTPU', input_text: str, tokenizer,
                search_method: str = "bfs", max_thoughts: int = 5,
                temperature: float = 0.8, max_steps: int = 50) -> str:
    """
    Generate solution using Tree of Thoughts with automatic search method selection.
    
    Args:
        model: ToTModelTPU instance
        input_text: Input text prompt
        tokenizer: Tokenizer instance
        search_method: Search strategy ("bfs" or "dfs")
        max_thoughts: Maximum number of thoughts to consider at each step
        temperature: Sampling temperature for thought generation
        max_steps: Maximum number of steps to search
        
    Returns:
        Generated solution with thought process
    """
    # Ensure model parameters are set
    model.max_thoughts = max_thoughts
    model.max_steps = max_steps
    
    # Generate solution using Tree of Thoughts
    solution = model.solve_with_tot(
        input_text=input_text,
        tokenizer=tokenizer,
        search_method=search_method,
        b=max_thoughts
    )
    
    return solution

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
            "thought_text": self.thought_text,
            "node_id": self.node_id,
            "score": float(self.score),
            "depth": self.depth,
            "hidden_state": self.hidden_state.tolist() if self.hidden_state is not None else None
        }

    def path_to_root(self) -> List[str]:
        """Return path from this node to root"""
        path = [self.thought_text]
        current = self
        while current.parent:
            current = current.parent
            path.append(current.thought_text)
        return path[::-1]

class ToTModelTPU(CoTModelTPU):
    """Tree of Thoughts model with TPU optimization"""
    
    def __init__(self, embed_dim=512, num_layers=12, num_heads=8, ff_dim=2048,
                 vocab_size=50000, max_seq_len=512, num_experts=7, 
                 max_thoughts=5, max_depth=10, max_steps=50,
                 candidates_per_step=10, dropout_rate=0.1,
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
        self.max_steps = max_steps
        self.candidates_per_step = candidates_per_step
        self.eval_head = TPUGEMMLinear(3)  # 3 classes for evaluation
        self.thought_generator = TPUGEMMLinear(vocab_size)

    def __call__(self, input_ids: jnp.ndarray, target_ids: Optional[jnp.ndarray] = None,
                 is_training: bool = True) -> Tuple[jnp.ndarray, Optional[jnp.ndarray]]:
        # Call parent class method to get base transformer output
        logits, base_loss = super().__call__(input_ids, target_ids, is_training)

        # Add ToT-specific processing if needed
        if is_training and target_ids is not None:
            # Additional ToT-specific loss calculation
            thought_loss = self._compute_thought_loss(logits, target_ids)
            total_loss = base_loss + thought_loss if base_loss is not None else thought_loss
            return logits, total_loss
        
        return logits, None

    def _compute_thought_loss(self, logits: jnp.ndarray, target_ids: jnp.ndarray) -> jnp.ndarray:
        """Compute ToT-specific loss component"""
        # Get final hidden state
        hidden = logits[:, -1, :]
        # Pass through evaluation head
        eval_logits = self.eval_head(hidden)
        # Convert target to one-hot
        target_classes = jnp.zeros(eval_logits.shape[0], dtype=jnp.int32)
        target_one_hot = jax.nn.one_hot(target_classes, 3)
        # Compute cross entropy loss
        thought_loss = optax.softmax_cross_entropy(eval_logits, target_one_hot)
        return jnp.mean(thought_loss)

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
        # Encode input context
        input_ids = tokenizer.encode(input_text, return_tensors="jax")
        current_ids = tokenizer.encode(current_thought, return_tensors="jax")
        combined_ids = jnp.concatenate([input_ids, current_ids], axis=1)

        # Get logits for next tokens
        logits, _ = self(combined_ids, is_training=False)
        next_token_logits = logits[:, -1, :]

        # Sample candidates
        candidates = []
        rng = jax.random.PRNGKey(0)  # Should be managed better in practice
        for i in range(num_candidates):
            rng, sample_rng = jax.random.split(rng)
            sample = jax.random.categorical(sample_rng, next_token_logits / 0.8)
            decoded = tokenizer.decode(sample)
            candidates.append(f"{current_thought} -> {decoded}")

        return candidates

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

                        if score > best_score and "=" in cand_text:
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

                    if score > best_score and "=" in cand_text:
                        best_node = child_node
                        best_score = score

                # Sort and add best candidates to stack
                child_nodes.sort(key=lambda x: x.score, reverse=True)
                stack.extend((node, step + 1) for node in child_nodes[:b])

        return best_node

# Example usage and test
if __name__ == "__main__":
    def run_tot(x: jnp.ndarray) -> jnp.ndarray:
        model = ToTModelTPU()
        return model(x, is_training=False)[0]

    # Initialize
    batch_size, seq_len = 2, 64
    rng = jax.random.PRNGKey(0)
    input_ids = jax.random.randint(rng, (batch_size, seq_len), 0, 50000, dtype=jnp.int32)

    # Transform and initialize
    transformed = hk.transform(run_tot)
    params = transformed.init(rng, input_ids)

    # Forward pass
    logits = transformed.apply(params, rng, input_ids)
    print("ToT Output shape:", logits.shape)