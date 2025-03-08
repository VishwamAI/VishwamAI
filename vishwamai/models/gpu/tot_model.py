# /home/kasinadhsarma/VishwamAI/vishwamai/models/tot_model.py
"""
Tree of Thoughts (ToT) model for VishwamAI, extending the CoT model with tree search.
Supports DFS and BFS for thought exploration, thought generation, and evaluation.
Designed for complex reasoning tasks requiring deep calculations.
"""

import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import deque
import heapq

import os
import time
from typing import Dict, List, Optional, Tuple, Any
from abc import ABC, abstractmethod
import torch
import torch.nn as nn
from torch.nn import functional as F

# Update import to use the new VishwamAI Transformer
from vishwamai.models.transformer import VishwamAITransformer
from vishwamai.models.attention import OptimizedMoEAttention
from vishwamai.models.cot_model import CoTModel, extract_answer
from vishwamai.models.gpu.optimizations.deep_ep import Buffer

class ThoughtNode(ABC):
    """
    Abstract base class representing a node in the thought tree with 3FS state management.
    """
    def __init__(
        self,
        thought_text: str,
        node_id: str,
        parent=None,
        score: float = 0.0,
        hidden_state: Optional[torch.Tensor] = None
    ):
        self.thought_text = thought_text
        self.node_id = node_id
        self.parent = parent
        self.children = []
        self.score = score
        self.depth = parent.depth + 1 if parent else 0
        self.hidden_state = hidden_state
        
        # Track state changes
        self._is_dirty = True
        self._is_cached = False
        
    def add_child(self, child: 'ThoughtNode') -> None:
        self.children.append(child)
        self._is_dirty = True
        
    def get_state_dict(self) -> Dict[str, Any]:
        """Get node state for persistence"""
        return {
            'text': self.thought_text,
            'node_id': self.node_id,
            'parent_id': self.parent.node_id if self.parent else None,
            'score': self.score,
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
        
    @property
    def needs_sync(self) -> bool:
        """Check if node needs to be synced to storage"""
        return self._is_dirty and not self._is_cached

class OptimizedMoELayer:
    """Optimized MoE layer using DeepEP for efficient expert parallelism"""
    def __init__(self, num_experts, hidden_dim, group=None):
        self._buffer = None
        self.num_experts = num_experts
        self.hidden_dim = hidden_dim
        self.group = group
        
        # Set optimal SM count for dispatch/combine
        Buffer.set_num_sms(24)  # Can be tuned based on hardware
        
    def _get_buffer(self):
        if self._buffer is None:
            self._buffer = Buffer(
                self.group,
                hidden_bytes=self.hidden_dim * 2,  # For bf16
                num_nvl_bytes=0,  # Will be calculated by the library
                num_rdma_bytes=0  # Will be calculated by the library
            )
        return self._buffer
        
    def dispatch(self, x, topk_idx, topk_weights, prev_event=None):
        """Dispatch tokens to experts using DeepEP's optimized kernel"""
        buffer = self._get_buffer()
        
        # Get dispatch layout
        num_tokens_per_rank, num_tokens_rdma, num_tokens_per_expert, is_token_in_rank, event = \
            buffer.get_dispatch_layout(
                topk_idx, self.num_experts,
                previous_event=prev_event,
                async_finish=True,
                allocate_on_comm_stream=prev_event is not None
            )
            
        # Perform optimized dispatch
        recv_x, recv_idx, recv_weights, expert_counts, handle, event = \
            buffer.dispatch(
                x, topk_idx=topk_idx,
                topk_weights=topk_weights, 
                num_tokens_per_rank=num_tokens_per_rank,
                num_tokens_per_rdma_rank=num_tokens_rdma,
                is_token_in_rank=is_token_in_rank,
                num_tokens_per_expert=num_tokens_per_expert,
                previous_event=prev_event,
                async_finish=True,
                allocate_on_comm_stream=True
            )
            
        return recv_x, recv_idx, recv_weights, expert_counts, handle, event
        
    def combine(self, expert_outputs, handle, topk_weights=None, prev_event=None):
        """Combine expert outputs using DeepEP's optimized kernel"""
        buffer = self._get_buffer()
        
        combined_output, _, event = buffer.combine(
            expert_outputs,
            handle,
            topk_weights=topk_weights,
            async_finish=True,
            previous_event=prev_event,
            allocate_on_comm_stream=prev_event is not None
        )
        
        return combined_output, event

class ToTModel(CoTModel):
    """
    Tree of Thoughts model with 3FS integration for optimized state management.
    """
    def __init__(
        self,
        embed_dim=512,
        num_layers=12,
        num_heads=8,
        ff_dim=2048,
        vocab_size=50000,
        max_seq_len=512,
        num_experts=7,
        max_steps=3,
        candidates_per_step=5,
        max_depth=10,
        use_3fs=True,
        cache_dir="/tmp/vishwamai/tot_cache"
    ):
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
        super(ToTModel, self).__init__(
            embed_dim=embed_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            ff_dim=ff_dim,
            vocab_size=vocab_size,
            max_seq_len=max_seq_len,
            num_experts=num_experts
        )
        self.max_steps = max_steps
        self.candidates_per_step = candidates_per_step
        self.max_depth = max_depth
        self.use_3fs = use_3fs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Optimized MoE layer
        if torch.cuda.is_available():
            self.moe_layer = OptimizedMoELayer(
                num_experts=num_experts,
                hidden_dim=embed_dim,
                group=None
            )

        # Initialize 3FS components
        if use_3fs:
            from vishwamai.models.gpu.integrations.tree_state_manager import TreeStateManager
            from vishwamai.models.gpu.integrations.kvcache_manager import KVCacheManager
            from vishwamai.models.gpu.integrations.state_persistence import StateManager
            
            # Tree state management
            self.tree_manager = TreeStateManager(
                storage_dir=os.path.join(cache_dir, "trees"),
                embed_dim=embed_dim
            )
            
            # KV Cache for thought generation
            self.kvcache = KVCacheManager(
                cache_dir=os.path.join(cache_dir, "kvcache"),
                embed_dim=embed_dim,
                num_heads=num_heads
            )
            
            # State persistence for model components
            self.state_manager = StateManager(
                os.path.join(cache_dir, "model_state"),
                embed_dim
            )
        else:
            self.tree_manager = None
            self.kvcache = None
            self.state_manager = None

        # Evaluation head using DeepGEMM
        from vishwamai.models.gpu.kernel_layers import DeepGEMMLinear
        self.eval_head = DeepGEMMLinear(
            embed_dim,
            3,  # sure/maybe/impossible
            use_3fs=use_3fs,
            cache_dir=os.path.join(cache_dir, "eval_head")
        )

    def evaluate_thought(self, thought_ids, batch_idx=0, seq_idx=0):
        """
        Evaluate a thought by predicting its likelihood of leading to a solution.
        
        Args:
            thought_ids (torch.Tensor): Token IDs of the thought (batch_size, seq_len).
        
        Returns:
            float: Score (probability of "sure").
        """
        # Try to get cached evaluation
        if self.use_3fs and self.kvcache is not None:
            cached_result = self.kvcache.retrieve(batch_idx, seq_idx)
            if cached_result is not None:
                return cached_result[0].item()
        
        with torch.no_grad():
            # Get hidden state
            last_hidden = self.transformer.get_hidden_state(thought_ids)[:, -1, :]
            
            # Evaluate using optimized head
            eval_logits = self.eval_head(last_hidden)
            probs = F.softmax(eval_logits, dim=-1)
            sure_prob = probs[:, 0].item()
            
            # Cache result
            if self.use_3fs and self.kvcache is not None:
                self.kvcache.store(
                    torch.tensor([sure_prob]),
                    torch.tensor([sure_prob]),
                    batch_idx,
                    seq_idx
                )
                
        return sure_prob

    def generate_candidates(self, input_text, current_thought, tokenizer, num_candidates, tree_id=None):
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
        # Try to get cached candidates
        if self.use_3fs and self.tree_manager is not None and tree_id is not None:
            cached_paths = self.tree_manager.get_cached_paths(tree_id, min_score=0.7)
            if cached_paths:
                # Use previous successful paths to guide candidate generation
                best_path = cached_paths[0][0]
                current_idx = best_path.index(current_thought) if current_thought in best_path else -1
                if current_idx >= 0 and current_idx < len(best_path) - 1:
                    candidates = [best_path[current_idx + 1]]
                    num_candidates -= 1  # Generate one less new candidate
        
        # Construct generation prompt
        prompt = f"{input_text}\nCurrent thought: {current_thought}\nPropose next steps:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)
        
        # Generate new candidates
        candidates = []
        for i in range(num_candidates):
            output_ids = self._sample(
                input_ids,
                max_length=50,
                temperature=0.8,
                top_p=0.9,
                batch_idx=hash(tree_id) if tree_id else 0,
                seq_idx=i
            )
            candidate_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
            candidate_text = candidate_text.replace(prompt, "").strip()
            if candidate_text:
                candidates.append(candidate_text)
                
                # Cache generated candidate if using 3FS
                if self.use_3fs and self.tree_manager is not None and tree_id is not None:
                    node_id = f"{tree_id}_candidate_{i}"
                    self.tree_manager.store_node(
                        node_id,
                        {'text': candidate_text},
                        hidden_state=self.transformer.get_hidden_state(output_ids)
                    )
        
        return candidates[:num_candidates]

    def solve_with_tot(
        self,
        input_text: str,
        tokenizer,
        search_method: str = "bfs",
        b: int = 5,
        tree_id: Optional[str] = None
    ):
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
        # Generate unique tree ID if not provided
        if tree_id is None:
            tree_id = f"tree_{hash(input_text)}_{int(time.time())}"
            
        # Try to load existing tree state
        if self.use_3fs and self.tree_manager is not None:
            tree_data = self.tree_manager.load_tree_structure(tree_id)
            if tree_data is not None:
                # Resume from saved state
                root = self._reconstruct_tree(tree_data, tree_id)
            else:
                # Create new tree
                root = ThoughtNode("Start", node_id=f"{tree_id}_root", score=1.0)
                self._save_tree_state(tree_id, root)
        else:
            root = ThoughtNode("Start", node_id=f"{tree_id}_root", score=1.0)
            
        # Perform tree search
        if search_method.lower() == "bfs":
            final_node = self._bfs_search(input_text, root, tokenizer, b, tree_id)
        else:
            final_node = self._dfs_search(input_text, root, tokenizer, b, tree_id)
            
        if final_node:
            # Get solution path
            thought_path = final_node.path_to_root()[1:]
            thought_text = " -> ".join(thought_path)
            answer = thought_path[-1].split("=")[-1].strip() if "=" in thought_path[-1] else "No solution found"
            
            # Cache successful path
            if self.use_3fs and self.tree_manager is not None:
                self.tree_manager.store_path(tree_id, thought_path, final_node.score)
                
            return f"<think>{thought_text}</think> <answer>{answer}</answer>"
        else:
            return "<think>Failed to find a solution.</think> <answer>No solution</answer>"
            
    def _save_tree_state(self, tree_id: str, root: ThoughtNode) -> None:
        """Save current tree state to 3FS"""
        if not self.use_3fs or self.tree_manager is None:
            return
            
        # Collect all nodes
        nodes = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            nodes.append(node)
            queue.extend(node.children)
            
        # Save tree structure
        tree_data = {
            'root_id': root.node_id,
            'nodes': {node.node_id: node.get_state_dict() for node in nodes}
        }
        self.tree_manager.store_tree_structure(tree_id, tree_data)
        
        # Save node states that need syncing
        for node in nodes:
            if node.needs_sync:
                self.tree_manager.store_node(
                    node.node_id,
                    {'text': node.thought_text},
                    hidden_state=node.hidden_state
                )
                node._is_dirty = False
                node._is_cached = True
                
    def _reconstruct_tree(self, tree_data: Dict[str, Any], tree_id: str) -> ThoughtNode:
        """Reconstruct tree from saved state"""
        nodes = {}
        node_data = tree_data['nodes']
        
        # First pass: Create all nodes
        for node_id, data in node_data.items():
            node = ThoughtNode(
                thought_text=data['text'],
                node_id=node_id,
                score=data.get('score', 0.0)
            )
            nodes[node_id] = node
            
            # Load node state if available
            if self.use_3fs and self.tree_manager is not None:
                _, hidden_state, _ = self.tree_manager.load_node(node_id, load_hidden=True)
                if hidden_state is not None:
                    node.hidden_state = hidden_state
                    
        # Second pass: Connect nodes
        for node_id, data in node_data.items():
            node = nodes[node_id]
            if data['parent_id']:
                node.parent = nodes[data['parent_id']]
            for child_id in data['child_ids']:
                node.add_child(nodes[child_id])
                
        return nodes[tree_data['root_id']]

    def _bfs_search(self, input_text: str, root: ThoughtNode, tokenizer, b: int, tree_id: str):
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
        best_node = None
        best_score = -float('inf')

        while queue and step < self.max_steps:
            level_size = len(queue)
            level_nodes = []
            
            for _ in range(level_size):
                current_node = queue.popleft()
                if current_node.depth >= self.max_depth:
                    continue

                # Get candidates using previous solutions if available
                candidates = self.generate_candidates(
                    input_text,
                    current_node.thought_text,
                    tokenizer,
                    self.candidates_per_step,
                    tree_id
                )
                
                for candidate in candidates:
                    # Create and evaluate candidate node
                    node_id = f"{tree_id}_level{step}_cand{len(level_nodes)}"
                    candidate_ids = tokenizer.encode(candidate, return_tensors="pt").to(self.device)
                    
                    # Get cached score if available
                    score = self.evaluate_thought(
                        candidate_ids,
                        batch_idx=hash(node_id),
                        seq_idx=0
                    )
                    
                    # Create node with hidden state
                    hidden_state = self.transformer.get_hidden_state(candidate_ids)
                    child_node = ThoughtNode(
                        thought_text=candidate,
                        node_id=node_id,
                        parent=current_node,
                        score=score,
                        hidden_state=hidden_state
                    )
                    current_node.add_child(child_node)
                    level_nodes.append(child_node)
                    
                    # Update best node
                    if score > best_score and "24" in candidate and "=" in candidate:
                        best_node = child_node
                        best_score = score

            # Keep top b nodes
            level_nodes.sort(key=lambda x: x.score, reverse=True)
            queue.extend(level_nodes[:b])
            
            # Save current tree state
            if self.use_3fs and self.tree_manager is not None:
                self._save_tree_state(tree_id, root)
                
            step += 1
            
        return best_node

    def _dfs_search(self, input_text: str, root: ThoughtNode, tokenizer, b: int, tree_id: str):
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
        stack = [(root, 0)]
        best_node = None
        best_score = -float('inf')
        visited = set()

        while stack:
            current_node, step = stack.pop()
            if step >= self.max_steps or current_node.depth >= self.max_depth:
                continue
                
            # Skip if already visited
            if current_node.node_id in visited:
                continue
            visited.add(current_node.node_id)

            # Generate and evaluate candidates
            candidates = self.generate_candidates(
                input_text,
                current_node.thought_text,
                tokenizer,
                self.candidates_per_step,
                tree_id
            )
            
            scored_candidates = []
            for i, candidate in enumerate(candidates):
                node_id = f"{tree_id}_step{step}_cand{i}"
                candidate_ids = tokenizer.encode(candidate, return_tensors="pt").to(self.device)
                
                # Get score and hidden state
                score = self.evaluate_thought(
                    candidate_ids,
                    batch_idx=hash(node_id),
                    seq_idx=0
                )
                hidden_state = self.transformer.get_hidden_state(candidate_ids)
                
                # Create node
                child_node = ThoughtNode(
                    thought_text=candidate,
                    node_id=node_id,
                    parent=current_node,
                    score=score,
                    hidden_state=hidden_state
                )
                current_node.add_child(child_node)
                scored_candidates.append(child_node)
                
                # Update best solution
                if score > best_score and "24" in candidate and "=" in candidate:
                    best_node = child_node
                    best_score = score

            # Sort and select top candidates
            scored_candidates.sort(key=lambda x: x.score, reverse=True)
            for candidate in scored_candidates[:b]:
                stack.append((candidate, step + 1))
                
            # Save state periodically
            if self.use_3fs and self.tree_manager is not None:
                self._save_tree_state(tree_id, root)

        return best_node

    def _process_expert_outputs(self, hidden_states, expert_scores):
        # Use optimized MoE dispatch/combine if on GPU
        if torch.cuda.is_available():
            # Get top-k expert assignments
            topk_idx = expert_scores.topk(k=2, dim=-1)[1]
            topk_weights = F.softmax(expert_scores, dim=-1)
            
            # Dispatch to experts
            dispatched_states, _, weights, counts, handle, event = \
                self.moe_layer.dispatch(hidden_states, topk_idx, topk_weights)
                
            # Process with experts
            expert_outputs = self.process_with_experts(dispatched_states, counts)
            
            # Combine expert outputs
            combined_output, _ = self.moe_layer.combine(
                expert_outputs, handle, topk_weights=weights,
                prev_event=event
            )
            
            return combined_output
        else:
            # Fallback to base implementation
            return super()._process_expert_outputs(hidden_states, expert_scores)

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
