# /home/kasinadhsarma/VishwamAI/vishwamai/models/tot_model.py
"""
Device-agnostic Tree of Thoughts (ToT) model for VishwamAI.
Supports both GPU (PyTorch) and TPU (JAX) execution with unified interface.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import deque
import heapq

try:
    import jax
    import jax.numpy as jnp
    from jax import random, grad, jit, vmap
    import flax.linen as flax_nn
    HAS_JAX = True
except ImportError:
    HAS_JAX = False

from vishwamai.models.transformer import VishwamAITransformer
from vishwamai.models.attention import OptimizedMoEAttention, DeviceAgnosticModule
from vishwamai.models.cot_model import CoTModel, extract_answer

def get_device_type():
    """Determine the available device type."""
    if torch.cuda.is_available():
        return "gpu"
    elif HAS_JAX and len(jax.devices("tpu")) > 0:
        return "tpu"
    return "cpu"

class ThoughtNode:
    """
    Represents a node in the thought tree.
    """
    def __init__(self, thought_text, parent=None, score=0.0):
        self.thought_text = thought_text  # Text of the thought (e.g., "4 + 9 = 13")
        self.parent = parent              # Parent node
        self.children = []                # Child nodes (next thoughts)
        self.score = score                # Evaluation score (e.g., "sure" probability)
        self.depth = parent.depth + 1 if parent else 0  # Depth in the tree

    def add_child(self, child):
        self.children.append(child)

    def path_to_root(self):
        """Return the path from this node to the root as a list of thoughts."""
        path = []
        current = self
        while current:
            path.append(current.thought_text)
            current = current.parent
        return list(reversed(path))

class ToTModel(CoTModel, DeviceAgnosticModule):
    """
    Tree of Thoughts model extending CoTModel with tree search capabilities.
    """
    def __init__(self, embed_dim=512, num_layers=12, num_heads=8, ff_dim=2048, 
                 vocab_size=50000, max_seq_len=512, num_experts=7, 
                 max_steps=3, candidates_per_step=5, max_depth=10, force_device=None):
        """
        Initialize the ToT model.
        
        Args:
            embed_dim (int): Embedding dimension.
            num_layers (int): Number of transformer layers.
            num_heads (int): Number of attention heads.
            ff_dim (int): Feed-forward dimension.
            vocab_size (int): Vocabulary size.
            max_seq_len (int): Maximum sequence length.
            num_experts (int): Number of attention experts for MoE.
            max_steps (int): Maximum reasoning steps (e.g., 3 for Game of 24).
            candidates_per_step (int): Number of candidate thoughts per step (e.g., 5).
            max_depth (int): Maximum depth of the thought tree.
        """
        CoTModel.__init__(self, embed_dim, num_layers, num_heads, ff_dim, 
                         vocab_size, max_seq_len, num_experts, force_device)
        DeviceAgnosticModule.__init__(self)
        
        if force_device:
            self.device_type = force_device
        self.max_steps = max_steps
        self.candidates_per_step = candidates_per_step
        self.max_depth = max_depth
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Device-specific evaluation head
        if self.device_type == "tpu" and HAS_JAX:
            self.eval_head = flax_nn.Dense(3)  # JAX implementation
        else:
            self.eval_head = nn.Linear(embed_dim, 3)  # PyTorch implementation

    def evaluate_thought(self, thought_ids, use_cache=True):
        """
        Evaluate a thought by predicting its likelihood of leading to a solution.
        
        Args:
            thought_ids (torch.Tensor): Token IDs of the thought (batch_size, seq_len).
        
        Returns:
            float: Score (probability of "sure").
        """
        # Convert input to appropriate device format
        thought_ids = self.to_device(thought_ids)

        if self.device_type == "tpu" and HAS_JAX:
            # JAX/TPU implementation
            @jit
            def evaluate_fn(params, inputs):
                logits = self.transformer.apply({'params': params}, inputs)
                last_hidden = logits[:, -1, :]
                eval_logits = self.eval_head.apply({'params': params}, last_hidden)
                probs = jax.nn.softmax(eval_logits, axis=-1)
                return probs[:, 0]  # Return "sure" probability
            
            with jax.disable_jit() if not use_cache else jax.enable_jit():
                sure_prob = evaluate_fn(self.params, thought_ids)[0].item()
        else:
            # PyTorch/GPU implementation
            with torch.no_grad():
                logits = self.transformer(thought_ids)
                last_hidden = self.transformer.get_hidden_state(thought_ids, mask=None)[:, -1, :]
                eval_logits = self.eval_head(last_hidden)
                probs = F.softmax(eval_logits, dim=-1)
                sure_prob = probs[:, 0].item()
        
        return sure_prob

    def generate_candidates(self, input_text, current_thought, tokenizer, num_candidates, use_cache=True):
        """
        Generate candidate thoughts for the next step.
        
        Args:
            input_text (str): Original input problem.
            current_thought (str): Current thought text.
            tokenizer: Tokenizer instance.
            num_candidates (int): Number of candidates to generate.
        
        Returns:
            list: List of candidate thought texts.
        """
        # Construct prompt for thought generation
        prompt = f"{input_text}\nCurrent thought: {current_thought}\nPropose next steps:"
        
        # Handle different device types
        if self.device_type == "tpu" and HAS_JAX:
            input_ids = self.to_device(tokenizer.encode(prompt, return_tensors="jax"))
            
            @jit
            def generate_fn(params, inputs, rng_key):
                def sample_next_token(logits, rng_key):
                    temperature = 0.8
                    logits = logits / temperature
                    return random.categorical(rng_key, logits)
                
                # JAX generation loop
                output_ids = inputs
                rng_key = random.PRNGKey(0)
                
                for _ in range(50):  # max_length
                    logits = self.transformer.apply({'params': params}, output_ids)
                    next_token = sample_next_token(logits[:, -1, :], rng_key)
                    output_ids = jnp.concatenate([output_ids, next_token[:, None]], axis=1)
                    
                return output_ids
                
            with jax.disable_jit() if not use_cache else jax.enable_jit():
                candidates = []
                for i in range(num_candidates):
                    rng_key = random.PRNGKey(i)
                    output_ids = generate_fn(self.params, input_ids, rng_key)
                    candidate_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                    candidate_text = candidate_text.replace(prompt, "").strip()
                    if candidate_text:
                        candidates.append(candidate_text)
        else:
            # PyTorch/GPU implementation
            input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            candidates = []
            for _ in range(num_candidates):
                output_ids = self._sample(input_ids, max_length=50, temperature=0.8, top_p=0.9)
                candidate_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                candidate_text = candidate_text.replace(prompt, "").strip()
                if candidate_text:
                    candidates.append(candidate_text)
        
        return candidates[:num_candidates]

    def solve_with_tot(self, input_text, tokenizer, search_method="bfs", b=5, use_cache=True):
        """
        Solve the problem using Tree of Thoughts with BFS or DFS.
        
        Args:
            input_text (str): Input problem (e.g., "Game of 24: 4 9 10 13").
            tokenizer: Tokenizer instance.
            search_method (str): "bfs" or "dfs" for search strategy.
            b (int): Number of best candidates to keep at each step.
        
        Returns:
            str: Final output with <think> and <answer> tags.
        """
        # Initialize the thought tree
        root = ThoughtNode(thought_text="Start", score=1.0)
        if search_method.lower() == "bfs":
            final_node = self._bfs_search(input_text, root, tokenizer, b)
        else:
            final_node = self._dfs_search(input_text, root, tokenizer, b)
        
        # Construct the final output
        if final_node:
            thought_path = final_node.path_to_root()[1:]  # Exclude "Start"
            thought_text = " -> ".join(thought_path)
            answer = thought_path[-1].split("=")[-1].strip() if "=" in thought_path[-1] else "No solution found"
            return f"<think>{thought_text}</think> <answer>{answer}</answer>"
        else:
            return "<think>Failed to find a solution.</think> <answer>No solution</answer>"

    def _bfs_search(self, input_text, root, tokenizer, b):
        """
        Perform BFS to explore the thought tree.
        
        Args:
            input_text (str): Input problem.
            root (ThoughtNode): Root node of the thought tree.
            tokenizer: Tokenizer instance.
            b (int): Number of best candidates to keep at each step.
        
        Returns:
            ThoughtNode: Node with the final solution, or None.
        """
        queue = deque([root])
        step = 0

        while queue and step < self.max_steps:
            level_size = len(queue)
            for _ in range(level_size):
                current_node = queue.popleft()
                if current_node.depth >= self.max_depth:
                    continue

                # Generate candidate thoughts
                candidates = self.generate_candidates(input_text, current_node.thought_text, 
                                                    tokenizer, self.candidates_per_step)
                
                # Evaluate and add candidates to the tree
                for candidate in candidates:
                    candidate_ids = tokenizer.encode(candidate, return_tensors="pt").to(self.device)
                    score = self.evaluate_thought(candidate_ids)
                    child_node = ThoughtNode(thought_text=candidate, parent=current_node, score=score)
                    current_node.add_child(child_node)
                    queue.append(child_node)

            # Keep the top b candidates based on score
            if queue:
                scored_nodes = [(node, node.score) for node in queue]
                scored_nodes.sort(key=lambda x: x[1], reverse=True)
                queue = deque([node for node, _ in scored_nodes[:b]])

            step += 1

        # Find the best leaf node (highest score)
        if queue:
            best_node = max(queue, key=lambda x: x.score)
            # Check if the thought leads to the target (e.g., 24 in Game of 24)
            if "24" in best_node.thought_text and "=" in best_node.thought_text:
                return best_node
        return None

    def _dfs_search(self, input_text, root, tokenizer, b):
        """
        Perform DFS to explore the thought tree.
        
        Args:
            input_text (str): Input problem.
            root (ThoughtNode): Root node of the thought tree.
            tokenizer: Tokenizer instance.
            b (int): Number of best candidates to keep at each step.
        
        Returns:
            ThoughtNode: Node with the final solution, or None.
        """
        stack = [(root, 0)]  # (node, step)
        best_node = None
        best_score = -float("inf")

        while stack:
            current_node, step = stack.pop()
            if step >= self.max_steps or current_node.depth >= self.max_depth:
                continue

            # Generate candidate thoughts
            candidates = self.generate_candidates(input_text, current_node.thought_text, 
                                                tokenizer, self.candidates_per_step)
            
            # Evaluate candidates
            scored_candidates = []
            for candidate in candidates:
                candidate_ids = tokenizer.encode(candidate, return_tensors="pt").to(self.device)
                score = self.evaluate_thought(candidate_ids)
                child_node = ThoughtNode(thought_text=candidate, parent=current_node, score=score)
                current_node.add_child(child_node)
                scored_candidates.append(child_node)

            # Keep top b candidates
            scored_candidates.sort(key=lambda x: x.score, reverse=True)
            top_candidates = scored_candidates[:b]

            # Update best node if a solution is found
            for candidate in scored_candidates:
                if "24" in candidate.thought_text and "=" in candidate.thought_text:
                    if candidate.score > best_score:
                        best_node = candidate
                        best_score = candidate.score

            # Add top candidates to the stack
            for candidate in top_candidates:
                stack.append((candidate, step + 1))

        return best_node

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
