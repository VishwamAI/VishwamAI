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
import os
import time
from typing import Dict, List, Optional

# Import VishwamAI components
from vishwamai.models.transformer import VishwamAITransformer
from vishwamai.models.gpu.cot_model import CoTModel, extract_answer
from vishwamai.models.gpu.optimizations.deep_ep import Buffer
from vishwamai.models.gpu.kernel_layers import DeepGEMMLinear

class ThoughtNode:
    """Represents a node in the thought tree with 3FS state management"""
    def __init__(self, thought_text: str, node_id: str, parent=None, score: float = 0.0,
                 hidden_state: Optional[torch.Tensor] = None):
        self.thought_text = thought_text
        self.node_id = node_id
        self.parent = parent
        self.children = []
        self.score = score
        self.depth = parent.depth + 1 if parent else 0
        self.hidden_state = hidden_state
        self._is_dirty = True
        self._is_cached = False

    def add_child(self, child: 'ThoughtNode') -> None:
        self.children.append(child)
        self._is_dirty = True

    def get_state_dict(self) -> Dict[str, any]:
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
        Buffer.set_num_sms(24)  # Tunable based on hardware

    def _get_buffer(self):
        if self._buffer is None:
            self._buffer = Buffer(
                self.group,
                hidden_bytes=self.hidden_dim * 2,  # For bf16
                num_nvl_bytes=0,
                num_rdma_bytes=0
            )
        return self._buffer

    def dispatch(self, x, topk_idx, topk_weights, prev_event=None):
        """Dispatch tokens to experts using DeepEP's optimized kernel"""
        buffer = self._get_buffer()
        num_tokens_per_rank, num_tokens_rdma, num_tokens_per_expert, is_token_in_rank, event = \
            buffer.get_dispatch_layout(
                topk_idx, self.num_experts,
                previous_event=prev_event,
                async_finish=True,
                allocate_on_comm_stream=prev_event is not None
            )
        recv_x, recv_idx, recv_weights, expert_counts, handle, event = \
            buffer.dispatch(
                x, topk_idx=topk_idx, topk_weights=topk_weights,
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
    """Tree of Thoughts model with 3FS integration for optimized state management"""
    def __init__(self, embed_dim=512, num_layers=12, num_heads=8, ff_dim=2048,
                 vocab_size=50000, max_seq_len=512, num_experts=7, max_steps=3,
                 candidates_per_step=5, max_depth=10, use_3fs=True,
                 cache_dir="/tmp/vishwamai/tot_cache"):
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

        if torch.cuda.is_available():
            self.moe_layer = OptimizedMoELayer(num_experts=num_experts, hidden_dim=embed_dim)

        if use_3fs:
            from vishwamai.models.gpu.integrations.tree_state_manager import TreeStateManager
            from vishwamai.models.gpu.integrations.kvcache_manager import KVCacheManager
            from vishwamai.models.gpu.integrations.state_persistence import StateManager

            self.tree_manager = TreeStateManager(
                storage_dir=os.path.join(cache_dir, "trees"),
                embed_dim=embed_dim
            )
            self.kvcache = KVCacheManager(
                cache_dir=os.path.join(cache_dir, "kvcache"),
                embed_dim=embed_dim,
                num_heads=num_heads
            )
            self.state_manager = StateManager(
                os.path.join(cache_dir, "model_state"),
                embed_dim
            )
        else:
            self.tree_manager = None
            self.kvcache = None
            self.state_manager = None

        self.eval_head = DeepGEMMLinear(embed_dim, 3, use_3fs=use_3fs,
                                        cache_dir=os.path.join(cache_dir, "eval_head"))

    def evaluate_thought(self, thought_ids, batch_idx=0, seq_idx=0):
        """Evaluate a thought's likelihood of leading to a solution"""
        if self.use_3fs and self.kvcache is not None:
            cached_result = self.kvcache.retrieve(batch_idx, seq_idx)
            if cached_result is not None:
                return cached_result[0].item()

        with torch.no_grad():
            last_hidden = self.transformer.get_hidden_state(thought_ids)[:, -1, :]
            eval_logits = self.eval_head(last_hidden)
            probs = F.softmax(eval_logits, dim=-1)
            sure_prob = probs[:, 0].item()

            if self.use_3fs and self.kvcache is not None:
                self.kvcache.store(torch.tensor([sure_prob]), torch.tensor([sure_prob]),
                                   batch_idx, seq_idx)
        return sure_prob

    def generate_candidates(self, input_text, current_thought, tokenizer, num_candidates, tree_id=None):
        """Generate candidate thoughts for the next step"""
        candidates = []
        if self.use_3fs and self.tree_manager is not None and tree_id is not None:
            cached_paths = self.tree_manager.get_cached_paths(tree_id, min_score=0.7)
            if cached_paths:
                best_path = cached_paths[0][0]
                current_idx = best_path.index(current_thought) if current_thought in best_path else -1
                if current_idx >= 0 and current_idx < len(best_path) - 1:
                    candidates.append(best_path[current_idx + 1])
                    num_candidates -= 1

        prompt = f"{input_text}\nCurrent thought: {current_thought}\nPropose next steps:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(self.device)

        for i in range(num_candidates):
            output_ids = self._sample(input_ids, max_length=50, temperature=0.8, top_p=0.9)
            candidate_text = tokenizer.decode(output_ids[0], skip_special_tokens=True).replace(prompt, "").strip()
            if candidate_text and candidate_text not in candidates:
                candidates.append(candidate_text)
                if self.use_3fs and self.tree_manager is not None and tree_id is not None:
                    node_id = f"{tree_id}_candidate_{i}"
                    self.tree_manager.store_node(
                        node_id,
                        {'text': candidate_text},
                        hidden_state=self.transformer.get_hidden_state(output_ids)
                    )

        return candidates[:num_candidates]

    def solve_with_tot(self, input_text: str, tokenizer, search_method: str = "bfs", b: int = 5,
                       tree_id: Optional[str] = None):
        """Solve the problem using Tree of Thoughts with BFS or DFS"""
        if tree_id is None:
            tree_id = f"tree_{hash(input_text)}_{int(time.time())}"

        if self.use_3fs and self.tree_manager is not None:
            tree_data = self.tree_manager.load_tree_structure(tree_id)
            root = self._reconstruct_tree(tree_data, tree_id) if tree_data else \
                ThoughtNode("Start", node_id=f"{tree_id}_root", score=1.0)
            if tree_data is None:
                self._save_tree_state(tree_id, root)
        else:
            root = ThoughtNode("Start", node_id=f"{tree_id}_root", score=1.0)

        final_node = self._bfs_search(input_text, root, tokenizer, b, tree_id) if \
            search_method.lower() == "bfs" else self._dfs_search(input_text, root, tokenizer, b, tree_id)

        if final_node:
            thought_path = final_node.path_to_root()[1:]
            thought_text = " -> ".join(thought_path)
            answer = thought_path[-1].split("=")[-1].strip() if "=" in thought_path[-1] else "No solution found"
            if self.use_3fs and self.tree_manager is not None:
                self.tree_manager.store_path(tree_id, thought_path, final_node.score)
            return f"<think>{thought_text}</think> <answer>{answer}</answer>"
        return "<think>Failed to find a solution.</think> <answer>No solution</answer>"

    def _save_tree_state(self, tree_id: str, root: ThoughtNode) -> None:
        """Save current tree state to 3FS"""
        if not self.use_3fs or self.tree_manager is None:
            return

        nodes = []
        queue = deque([root])
        while queue:
            node = queue.popleft()
            nodes.append(node)
            queue.extend(node.children)

        tree_data = {
            'root_id': root.node_id,
            'nodes': {node.node_id: node.get_state_dict() for node in nodes}
        }
        self.tree_manager.store_tree_structure(tree_id, tree_data)

        for node in nodes:
            if node.needs_sync:
                self.tree_manager.store_node(
                    node.node_id,
                    {'text': node.thought_text},
                    hidden_state=node.hidden_state
                )
                node._is_dirty = False
                node._is_cached = True

    def _reconstruct_tree(self, tree_data: Dict[str, any], tree_id: str) -> ThoughtNode:
        """Reconstruct tree from saved state"""
        nodes = {}
        node_data = tree_data['nodes']

        for node_id, data in node_data.items():
            node = ThoughtNode(
                thought_text=data['text'],
                node_id=node_id,
                score=data.get('score', 0.0)
            )
            nodes[node_id] = node
            if self.use_3fs and self.tree_manager is not None:
                _, hidden_state, _ = self.tree_manager.load_node(node_id, load_hidden=True)
                if hidden_state is not None:
                    node.hidden_state = hidden_state

        for node_id, data in node_data.items():
            node = nodes[node_id]
            if data['parent_id']:
                node.parent = nodes[data['parent_id']]
            for child_id in data['child_ids']:
                node.add_child(nodes[child_id])

        return nodes[tree_data['root_id']]

    def _bfs_search(self, input_text: str, root: ThoughtNode, tokenizer, b: int, tree_id: str):
        """Perform BFS to explore the thought tree"""
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

                candidates = self.generate_candidates(input_text, current_node.thought_text,
                                                     tokenizer, self.candidates_per_step, tree_id)

                for candidate in candidates:
                    node_id = f"{tree_id}_level{step}_cand{len(level_nodes)}"
                    candidate_ids = tokenizer.encode(candidate, return_tensors="pt").to(self.device)
                    score = self.evaluate_thought(candidate_ids, batch_idx=hash(node_id), seq_idx=0)
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

                    if score > best_score and "24" in candidate and "=" in candidate:
                        best_node = child_node
                        best_score = score

            level_nodes.sort(key=lambda x: x.score, reverse=True)
            queue.extend(level_nodes[:b])
            if self.use_3fs and self.tree_manager is not None:
                self._save_tree_state(tree_id, root)
            step += 1

        return best_node

    def _dfs_search(self, input_text: str, root: ThoughtNode, tokenizer, b: int, tree_id: str):
        """Perform DFS to explore the thought tree"""
        stack = [(root, 0)]
        best_node = None
        best_score = -float('inf')
        visited = set()

        while stack:
            current_node, step = stack.pop()
            if step >= self.max_steps or current_node.depth >= self.max_depth or current_node.node_id in visited:
                continue
            visited.add(current_node.node_id)

            candidates = self.generate_candidates(input_text, current_node.thought_text,
                                                 tokenizer, self.candidates_per_step, tree_id)

            scored_candidates = []
            for i, candidate in enumerate(candidates):
                node_id = f"{tree_id}_step{step}_cand{i}"
                candidate_ids = tokenizer.encode(candidate, return_tensors="pt").to(self.device)
                score = self.evaluate_thought(candidate_ids, batch_idx=hash(node_id), seq_idx=0)
                hidden_state = self.transformer.get_hidden_state(candidate_ids)
                child_node = ThoughtNode(
                    thought_text=candidate,
                    node_id=node_id,
                    parent=current_node,
                    score=score,
                    hidden_state=hidden_state
                )
                current_node.add_child(child_node)
                scored_candidates.append(child_node)

                if score > best_score and "24" in candidate and "=" in candidate:
                    best_node = child_node
                    best_score = score

            scored_candidates.sort(key=lambda x: x.score, reverse=True)
            stack.extend((candidate, step + 1) for candidate in scored_candidates[:b])
            if self.use_3fs and self.tree_manager is not None:
                self._save_tree_state(tree_id, root)

        return best_node

    def _process_expert_outputs(self, hidden_states, expert_scores):
        """Process expert outputs with optimized MoE"""
        if torch.cuda.is_available():
            topk_idx = expert_scores.topk(k=2, dim=-1)[1]
            topk_weights = F.softmax(expert_scores, dim=-1)
            dispatched_states, _, weights, counts, handle, event = \
                self.moe_layer.dispatch(hidden_states, topk_idx, topk_weights)
            expert_outputs = self.process_with_experts(dispatched_states, counts)
            combined_output, _ = self.moe_layer.combine(expert_outputs, handle, topk_weights=weights,
                                                        prev_event=event)
            return combined_output
        return super()._process_expert_outputs(hidden_states, expert_scores)

    def process_with_experts(self, dispatched_states, counts):
        """Placeholder for expert processing"""
        # Simplified: assume identity transformation (actual implementation depends on experts)
        return dispatched_states

# Example usage
if __name__ == "__main__":
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
            tokens = [self.special_tokens.get(text, i) for i in range(5)]
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

    tokenizer = MockTokenizer()
    model = ToTModel(vocab_size=tokenizer.vocab_size)

    input_text = "Game of 24: 4 9 10 13"
    output_bfs = model.solve_with_tot(input_text, tokenizer, search_method="bfs", b=5)
    print("BFS Output:", output_bfs)
    print("BFS Extracted Answer:", extract_answer(output_bfs))

    output_dfs = model.solve_with_tot(input_text, tokenizer, search_method="dfs", b=5)
    print("DFS Output:", output_dfs)
    print("DFS Extracted Answer:", extract_answer(output_dfs))