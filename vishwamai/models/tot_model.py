# /home/kasinadhsarma/VishwamAI/vishwamai/models/tot_model.py
"""
Tree of Thoughts (ToT) model with device-agnostic implementation.
"""

from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F
from vishwamai.models.cot_model import CoTModel, DeviceAgnosticModule, extract_answer
import os

try:
    import jax
    import jax.numpy as jnp
    from jax import random, lax
    import flax.linen as nn
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

class ThoughtNode:
    """Node in the thought tree."""
    def __init__(self, thought_text: str, parent=None, score: float = 0.0):
        self.thought_text = thought_text
        self.parent = parent
        self.children = []
        self.score = score
        self.depth = 0 if parent is None else parent.depth + 1
        
    def add_child(self, child):
        self.children.append(child)
        
    def path_to_root(self) -> List[str]:
        path = [self.thought_text]
        current = self
        while current.parent:
            current = current.parent
            path.append(current.thought_text)
        return path[::-1]

class ToTModel(CoTModel):
    """Tree of Thoughts model with hardware-specific optimizations."""
    def __init__(
        self,
        embed_dim: int = 512,
        num_layers: int = 12,
        num_heads: int = 8,
        ff_dim: int = 2048,
        vocab_size: int = 50000,
        max_seq_len: int = 512,
        num_experts: int = 7,
        max_steps: int = 3,
        candidates_per_step: int = 5,
        max_depth: int = 10,
        force_device: str = None,
        cache_dir: str = "/tmp/vishwamai/tot_cache"
    ):
        super().__init__(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_experts=num_experts,
            force_device=force_device
        )
        
        self.max_steps = max_steps
        self.candidates_per_step = candidates_per_step
        self.max_depth = max_depth
        
        # Add evaluation head
        if self.device_type == "tpu" and HAS_JAX:
            self.eval_head = nn.Dense(3)  # sure/maybe/impossible
        else:
            import torch.nn as nn
            self.eval_head = nn.Linear(embed_dim, 3)
            
        # Initialize device-specific caching
        self._init_caching(cache_dir)
        
    def _init_caching(self, cache_dir: str):
        """Initialize device-specific caching."""
        if self.device_type == "gpu":
            from vishwamai.models.gpu.integrations.tree_state_manager import TreeStateManager
            from vishwamai.models.gpu.integrations.kvcache_manager import KVCacheManager
            
            # Tree state management
            self.tree_manager = TreeStateManager(
                storage_dir=os.path.join(cache_dir, "trees"),
                embed_dim=self.transformer.embed_dim
            )
            
            # KV Cache for thought generation
            self.kvcache = KVCacheManager(
                cache_dir=os.path.join(cache_dir, "kvcache"),
                embed_dim=self.transformer.embed_dim,
                num_heads=self.transformer.num_heads
            )
        else:
            self.tree_manager = None
            self.kvcache = None

    def evaluate_thought(self, thought_ids, training: bool = False) -> float:
        """Evaluate thought likelihood with device-specific implementation."""
        if self.device_type == "tpu" and HAS_JAX:
            logits = self.transformer(thought_ids, training=training)
            last_hidden = self.transformer.get_hidden_state(
                thought_ids, training=training
            )[:, -1, :]
            eval_logits = self.eval_head(last_hidden)
            probs = jax.nn.softmax(eval_logits, axis=-1)
            return probs[:, 0].mean()
        else:
            import torch
            import torch.nn.functional as F
            
            with torch.no_grad():
                logits = self.transformer(thought_ids)
                last_hidden = logits[:, -1, :]
                eval_logits = self.eval_head(last_hidden)
                probs = F.softmax(eval_logits, dim=-1)
                return probs[:, 0].mean().item()

    def generate_candidates(self, input_text: str, current_thought: str,
                          tokenizer, num_candidates: int, rng=None) -> List[str]:
        """Generate candidate thoughts with device-specific implementation."""
        prompt = f"{input_text}\nCurrent thought: {current_thought}\nPropose next steps:"
        
        if self.device_type == "tpu" and HAS_JAX:
            if rng is None:
                rng = random.PRNGKey(0)
                
            input_ids = jnp.array(
                tokenizer.encode(prompt, return_tensors="jax")[0],
                dtype=jnp.int32
            )
            
            def body_fn(val):
                i, candidates, rng = val
                if i >= num_candidates:
                    return i, candidates, rng
                    
                output_ids = self._sample_tpu(
                    input_ids[None, :], 50, 0.8, 0.9, rng
                )
                candidate_text = tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                )
                candidate_text = candidate_text.replace(prompt, "").strip()
                
                if candidate_text:
                    candidates = candidates.at[i].set(candidate_text)
                rng, sub_rng = random.split(rng)
                return i + 1, candidates, sub_rng

            candidates = jnp.array([""] * num_candidates)
            _, candidates, _ = lax.while_loop(
                lambda val: val[0] < num_candidates,
                body_fn,
                (0, candidates, rng)
            )
            return [c for c in candidates if c]
        else:
            import torch
            
            if rng is not None:
                torch.manual_seed(rng)
                
            input_ids = tokenizer.encode(
                prompt, return_tensors="pt"
            ).to(self.transformer.device)
            
            candidates = []
            for _ in range(num_candidates):
                output_ids = self._sample_gpu(input_ids, 50, 0.8, 0.9)
                candidate_text = tokenizer.decode(
                    output_ids[0], skip_special_tokens=True
                )
                candidate_text = candidate_text.replace(prompt, "").strip()
                if candidate_text:
                    candidates.append(candidate_text)
                    
            return candidates

    def _bfs_search(self, input_text: str, root: ThoughtNode, tokenizer,
                   b: int, rng=None) -> Optional[ThoughtNode]:
        """BFS with device-specific optimizations."""
        queue = [root]
        step = 0
        
        def process_candidates(node, candidates):
            new_nodes = []
            for candidate in candidates:
                if self.device_type == "tpu" and HAS_JAX:
                    candidate_ids = jnp.array(
                        tokenizer.encode(candidate, return_tensors="jax")[0],
                        dtype=jnp.int32
                    )[None, :]
                else:
                    import torch
                    candidate_ids = tokenizer.encode(
                        candidate, return_tensors="pt"
                    ).to(self.transformer.device)
                    
                score = self.evaluate_thought(candidate_ids)
                child = ThoughtNode(candidate, node, score)
                node.add_child(child)
                new_nodes.append(child)
            return new_nodes

        while queue and step < self.max_steps:
            current_nodes = queue
            new_queue = []
            
            for node in current_nodes:
                if node.depth >= self.max_depth:
                    continue
                    
                candidates = self.generate_candidates(
                    input_text, node.thought_text,
                    tokenizer, self.candidates_per_step, rng
                )
                
                new_nodes = process_candidates(node, candidates)
                new_queue.extend(new_nodes)
                
            if new_queue:
                # Keep top b candidates
                new_queue.sort(key=lambda x: x.score, reverse=True)
                queue = new_queue[:b]
            else:
                queue = []
                
            step += 1
            
        if queue:
            best_node = max(queue, key=lambda x: x.score)
            if "24" in best_node.thought_text and "=" in best_node.thought_text:
                return best_node
        return None

    def _dfs_search(self, input_text: str, root: ThoughtNode, tokenizer,
                   b: int, rng=None) -> Optional[ThoughtNode]:
        """DFS with device-specific optimizations."""
        stack = [(root, 0)]
        best_node = None
        best_score = float('-inf')
        
        while stack:
            current_node, step = stack.pop()
            
            if step >= self.max_steps or current_node.depth >= self.max_depth:
                continue
                
            candidates = self.generate_candidates(
                input_text, current_node.thought_text,
                tokenizer, self.candidates_per_step, rng
            )
            
            scored_candidates = []
            for candidate in candidates:
                if self.device_type == "tpu" and HAS_JAX:
                    candidate_ids = jnp.array(
                        tokenizer.encode(candidate, return_tensors="jax")[0],
                        dtype=jnp.int32
                    )[None, :]
                else:
                    import torch
                    candidate_ids = tokenizer.encode(
                        candidate, return_tensors="pt"
                    ).to(self.transformer.device)
                    
                score = self.evaluate_thought(candidate_ids)
                child = ThoughtNode(candidate, current_node, score)
                current_node.add_child(child)
                scored_candidates.append(child)
                
                if "24" in child.thought_text and "=" in child.thought_text:
                    if score > best_score:
                        best_node = child
                        best_score = score
                        
            # Keep top b candidates
            scored_candidates.sort(key=lambda x: x.score, reverse=True)
            stack.extend((c, step + 1) for c in scored_candidates[:b])
            
        return best_node

    def solve_with_tot(self, input_text: str, tokenizer, search_method: str = "bfs",
                      b: int = 5, rng=None) -> str:
        """Solve using ToT with device-specific optimizations."""
        if rng is None:
            if self.device_type == "tpu" and HAS_JAX:
                rng = random.PRNGKey(0)
            else:
                import torch
                rng = 0  # Will be used with torch.manual_seed
                
        # Initialize thought tree
        root = ThoughtNode(thought_text="Start", score=1.0)
        
        # Search using specified method
        if search_method.lower() == "bfs":
            final_node = self._bfs_search(input_text, root, tokenizer, b, rng)
        else:
            final_node = self._dfs_search(input_text, root, tokenizer, b, rng)
            
        # Construct output with thought path
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

        def encode(self, text, return_tensors="pt"):
            tokens = [self.special_tokens.get(text, i) for i in range(5)]  # Simplified
            if return_tensors == "pt":
                return torch.tensor([tokens], dtype=torch.long)
            return tokens

        def decode(self, token_ids, skip_special_tokens=False):
            text = ""
            for token in token_ids:
                if token.item() in self.inverse_vocab:
                    if not skip_special_tokens or token.item() < self.vocab_size-4:
                        text += self.inverse_vocab[token.item()] + " "
            return text.strip()

    # Initialize model and tokenizer
    tokenizer = MockTokenizer()
    model = ToTModel(vocab_size=tokenizer.vocab_size)

    # Example: Solve a Game of 24 problem
    input_text = "Game of 24: 4 9 10 13"
    output_bfs = model.solve_with_tot(input_text, tokenizer, search_method="bfs", b=5)
    print("BFS Output:", output_bfs)
    print("BFS Extracted Answer:", extract_answer(output_bfs))

    output_dfs = model.solve_with_tot(input_text, tokenizer, search_method="dfs", b=5)
    print("DFS Output:", output_dfs)
    print("DFS Extracted Answer:", extract_answer(output_dfs))
