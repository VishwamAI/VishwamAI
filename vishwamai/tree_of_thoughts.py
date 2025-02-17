import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Dict, Any, Union
from dataclasses import dataclass
from transformers import PreTrainedModel, GPT2LMHeadModel
import numpy as np
from torch.nn import functional as F
import math
from concurrent.futures import ThreadPoolExecutor
from torch.optim import Adam
from torch.distributions import Categorical

@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    math_reasoning_weight: float = 0.3
    logical_coherence_weight: float = 0.3
    real_world_applicability_weight: float = 0.2
    solution_validity_weight: float = 0.2
    
@dataclass
class TreeConfig:
    """Configuration for Tree of Thoughts implementation with training settings."""
    # Core parameters
    beam_width: int = 4
    max_depth: int = 3
    temperature: float = 0.7
    top_k: int = 50
    pruning_threshold: float = 0.1
    rewrite_factor: float = 0.3
    hidden_size: int = 8192
    memory_size: int = 1024
    num_memory_slots: int = 8
    uncertainty_threshold: float = 0.2
    merge_similarity_threshold: float = 0.85
    max_workers: int = 4
    
    # Training parameters
    learning_rate: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 10
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    reward_gamma: float = 0.99
    
    # Generation parameters
    max_length: int = 512
    min_length: int = 10
    num_beams: int = 4
    no_repeat_ngram_size: int = 3
    
    # Reasoning parameters
    max_math_steps: int = 8
    intermediate_steps: bool = True
    debug_mode: bool = False

class MathReasoningModule(nn.Module):
    """Module for mathematical reasoning and problem solving."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.operation_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 5)  # +, -, *, /, = operations
        )
        
        self.number_net = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, 1)
        )
        
        self.step_net = nn.GRUCell(hidden_size, hidden_size)
        
    def forward(self, x: torch.Tensor, num_steps: int) -> Tuple[torch.Tensor, List[Dict]]:
        steps = []
        current_state = x
        
        for _ in range(num_steps):
            # Predict operation and number
            op_logits = self.operation_net(current_state)
            num_value = self.number_net(current_state)
            
            # Store intermediate step
            steps.append({
                'operation': op_logits,
                'number': num_value,
                'state': current_state
            })
            
            # Update state
            current_state = self.step_net(current_state, current_state)
            
        return current_state, steps

class RewardFunction:
    """Compute rewards for reasoning steps."""
    def __init__(self, config: RewardConfig):
        self.config = config
        
    def compute_math_reasoning_reward(self, steps: List[Dict]) -> float:
        """Evaluate mathematical reasoning quality."""
        if not steps:
            return 0.0
            
        reward = 0.0
        prev_num = None
        
        for step in steps:
            op = torch.argmax(step['operation']).item()
            num = step['number'].item()
            
            if prev_num is not None:
                # Reward valid mathematical operations
                if op < 4:  # +, -, *, /
                    reward += 0.1
                # Reward reaching final answer
                if op == 4:  # =
                    reward += 0.3
                    
            prev_num = num
            
        return reward
        
    def compute_logical_coherence(self, states: List[torch.Tensor]) -> float:
        """Evaluate logical flow between reasoning steps."""
        if len(states) < 2:
            return 0.0
            
        coherence = 0.0
        for s1, s2 in zip(states[:-1], states[1:]):
            similarity = F.cosine_similarity(s1.view(1, -1), s2.view(1, -1)).item()
            coherence += similarity
            
        return coherence / (len(states) - 1)
        
    def __call__(self, node: 'TreeNode', steps: List[Dict]) -> float:
        """Compute overall reward for a reasoning path."""
        math_reward = self.compute_math_reasoning_reward(steps)
        coherence_reward = self.compute_logical_coherence([s['state'] for s in steps])
        
        # Combine rewards using weights
        total_reward = (
            self.config.math_reasoning_weight * math_reward +
            self.config.logical_coherence_weight * coherence_reward
        )
        
        return total_reward

class TreeNode:
    """Enhanced node with reasoning capabilities."""
    def __init__(self, state: torch.Tensor, score: float = 0.0):
        self.state = state
        self.score = score
        self.children: List[TreeNode] = []
        self.parent: Optional[TreeNode] = None
        self.depth: int = 0
        self.uncertainty: float = 1.0
        self.memory: Optional[torch.Tensor] = None
        self.reasoning_steps: List[Dict] = []
        self.text_output: Optional[str] = None
        
    def add_child(self, child: 'TreeNode'):
        """Add a child node and update its depth."""
        child.parent = self
        child.depth = self.depth + 1
        self.children.append(child)

class MemoryModule(nn.Module):
    """Memory-augmented reasoning module."""
    def __init__(self, hidden_size: int, memory_size: int, num_slots: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.memory_size = memory_size
        self.num_slots = num_slots
        
        self.query_net = nn.Linear(hidden_size, memory_size)
        self.key_net = nn.Linear(hidden_size, memory_size)
        self.value_net = nn.Linear(hidden_size, memory_size)
        
        self.memory = nn.Parameter(torch.randn(num_slots, memory_size))
        self.memory_keys = nn.Parameter(torch.randn(num_slots, memory_size))
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Generate query, key, value
        query = self.query_net(x)
        key = self.key_net(x)
        value = self.value_net(x)
        
        # Memory attention
        memory_attn = torch.matmul(query, self.memory_keys.T) / math.sqrt(self.memory_size)
        memory_attn = F.softmax(memory_attn, dim=-1)
        
        # Read from memory
        memory_read = torch.matmul(memory_attn, self.memory)
        
        # Update memory
        write_weights = F.softmax(torch.matmul(key, self.memory_keys.T), dim=-1)
        memory_update = torch.matmul(write_weights.T, value)
        self.memory.data = self.memory * 0.99 + memory_update * 0.01
        
        return memory_read, memory_attn

class TextGenerator(nn.Module):
    """Module for generating text from reasoning states."""
    def __init__(self, model_name: str = 'gpt2'):
        super().__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        
    def generate_text(self, state: torch.Tensor, config: TreeConfig) -> str:
        # Project state to model's hidden size
        proj = nn.Linear(state.size(-1), self.model.config.hidden_size)
        hidden_states = proj(state).unsqueeze(0)
        
        # Generate text
        output_ids = self.model.generate(
            inputs_embeds=hidden_states,
            max_length=config.max_length,
            min_length=config.min_length,
            num_beams=config.num_beams,
            no_repeat_ngram_size=config.no_repeat_ngram_size,
            temperature=config.temperature
        )
        
        return self.model.tokenizer.decode(output_ids[0], skip_special_tokens=True)

class TreeOfThoughts(nn.Module):
    """Enhanced Tree of Thoughts with training and advanced reasoning."""
    
    def __init__(self, 
                 model: PreTrainedModel,
                 config: Optional[TreeConfig] = None,
                 reward_config: Optional[RewardConfig] = None):
        super().__init__()
        self.config = config or TreeConfig()
        self.base_model = model
        
        # Components
        self.memory = MemoryModule(
            self.config.hidden_size,
            self.config.memory_size,
            self.config.num_memory_slots
        )
        self.executor = ThreadPoolExecutor(max_workers=self.config.max_workers)
        self.math_reasoning = MathReasoningModule(self.config.hidden_size)
        self.text_generator = TextGenerator()
        self.reward_function = RewardFunction(reward_config or RewardConfig())
        
        # Optimizer
        self.optimizer = Adam(self.parameters(), lr=self.config.learning_rate)
        
        # Reasoning networks
        self.state_evaluator = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, 1)
        )
        
        self.state_expander = nn.Sequential(
            nn.Linear(self.config.hidden_size, self.config.hidden_size * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size * self.config.beam_width)
        )
        
        self.state_refiner = nn.Sequential(
            nn.Linear(self.config.hidden_size * 2, self.config.hidden_size),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(self.config.hidden_size, self.config.hidden_size)
        )
        
    def _compute_uncertainty(self, state: torch.Tensor) -> float:
        """Compute uncertainty score for a state."""
        # Use dropout sampling for uncertainty estimation
        self.state_evaluator.train()
        samples = torch.stack([self.state_evaluator(state) for _ in range(5)])
        uncertainty = samples.std(dim=0).mean().item()
        self.state_evaluator.eval()
        return uncertainty

    def _merge_similar_states(self, nodes: List[TreeNode]) -> List[TreeNode]:
        """Merge nodes with similar states to prevent redundant paths."""
        if len(nodes) <= 1:
            return nodes
            
        merged = []
        used = set()
        
        for i, node1 in enumerate(nodes):
            if i in used:
                continue
                
            similar = [node1]
            for j, node2 in enumerate(nodes[i+1:], i+1):
                if j not in used:
                    similarity = F.cosine_similarity(
                        node1.state.view(1, -1),
                        node2.state.view(1, -1)
                    ).item()
                    if similarity > self.config.merge_similarity_threshold:
                        similar.append(node2)
                        used.add(j)
                        
            if len(similar) > 1:
                # Create merged node
                merged_state = torch.stack([n.state for n in similar]).mean(dim=0)
                merged_score = max(n.score for n in similar)
                merged_node = TreeNode(merged_state, merged_score)
                merged_node.uncertainty = min(n.uncertainty for n in similar)
                merged.append(merged_node)
            else:
                merged.append(node1)
                
        return merged

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Execute a single training step."""
        self.train()
        self.optimizer.zero_grad()
        
        # Forward pass with reasoning
        outputs = self(batch['input_ids'])
        
        # Compute rewards for each path
        rewards = []
        for node in self._get_leaf_nodes(outputs):
            reward = self.reward_function(node, node.reasoning_steps)
            rewards.append(reward)
            
        # Compute policy loss
        policy_loss = 0
        value_loss = 0
        
        for node, reward in zip(self._get_leaf_nodes(outputs), rewards):
            # Policy gradient loss
            log_probs = torch.stack([step['operation'].log_softmax(-1) for step in node.reasoning_steps])
            actions = torch.stack([step['operation'].argmax(-1) for step in node.reasoning_steps])
            policy_loss += -(log_probs * actions).sum() * reward
            
            # Value loss
            predicted_values = torch.stack([self.state_evaluator(step['state']) for step in node.reasoning_steps])
            value_targets = torch.tensor([reward * (self.config.reward_gamma ** i) 
                                       for i in range(len(node.reasoning_steps))])
            value_loss += F.mse_loss(predicted_values, value_targets)
        
        # Total loss
        loss = policy_loss + value_loss
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.parameters(), self.config.max_grad_norm)
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'policy_loss': policy_loss.item(),
            'value_loss': value_loss.item(),
            'mean_reward': sum(rewards) / len(rewards)
        }

    def forward(self, hidden_states: torch.Tensor) -> Union[torch.Tensor, List[TreeNode]]:
        """Process hidden states through advanced reasoning."""
        batch_size = hidden_states.size(0)
        device = hidden_states.device
        
        # Initialize root nodes for each batch item
        roots = [TreeNode(state) for state in hidden_states]
        
        # Build trees through iterative reasoning
        for depth in range(self.config.max_depth):
            level_nodes = [node for node in self._get_level_nodes(roots, depth)]
            
            if not level_nodes:
                break
                
            # Expand nodes
            expanded_states = []
            for node in level_nodes:
                expanded = self.state_expander(node.state)
                expanded = expanded.view(-1, self.config.beam_width, self.config.hidden_size)
                expanded_states.append(expanded)
            
            expanded_states = torch.cat(expanded_states, dim=0)
            
            # Mathematical reasoning
            math_states, math_steps = self.math_reasoning(
                expanded_states.view(-1, self.config.hidden_size),
                self.config.max_math_steps if self.config.intermediate_steps else 1
            )
            
            # Process with memory
            memory_reads, _ = self.memory(math_states)
            augmented_states = torch.cat([math_states, memory_reads], dim=-1)
            
            # Evaluate states with uncertainty
            scores = self.state_evaluator(augmented_states)
            scores = scores.view(-1, self.config.beam_width)
            
            # Select best children
            topk_scores, topk_indices = torch.topk(
                scores, k=min(self.config.beam_width, scores.size(1)), dim=1
            )
            
            # Create child nodes with advanced processing
            futures = []
            for i, node in enumerate(level_nodes):
                node_scores = topk_scores[i]
                node_states = math_states[i][topk_indices[i]]
                
                for score, state, steps in zip(node_scores, node_states, math_steps):
                    threshold = self.config.pruning_threshold * (1 + node.uncertainty)
                    
                    if score > threshold:
                        future = self.executor.submit(self._compute_uncertainty, state)
                        futures.append((node, state, score, steps, future))
            
            # Process results with reasoning steps
            for node, state, score, steps, future in futures:
                uncertainty = future.result()
                if uncertainty < self.config.uncertainty_threshold:
                    child = TreeNode(state, score.item())
                    child.uncertainty = uncertainty
                    child.memory = memory_reads[0]
                    child.reasoning_steps = steps
                    
                    # Generate text output for the reasoning step
                    if self.config.debug_mode:
                        child.text_output = self.text_generator.generate_text(state, self.config)
                    
                    node.add_child(child)
            
            # Merge similar nodes to prevent redundant paths
            for node in level_nodes:
                node.children = self._merge_similar_states(node.children)
        
        # Extract best paths and refine final states
        best_paths = self._extract_best_paths(roots)
        refined_states = self._refine_states(best_paths, hidden_states)
        
        return refined_states
    
    def _get_level_nodes(self, roots: List[TreeNode], depth: int) -> List[TreeNode]:
        """Get all nodes at a specific depth."""
        if depth == 0:
            return roots
            
        nodes = []
        def collect_nodes(node: TreeNode, current_depth: int):
            if current_depth == depth:
                nodes.append(node)
            else:
                for child in node.children:
                    collect_nodes(child, current_depth + 1)
                    
        for root in roots:
            collect_nodes(root, 0)
            
        return nodes
    
    def _extract_best_paths(self, roots: List[TreeNode]) -> List[List[torch.Tensor]]:
        """Extract best reasoning paths from trees."""
        paths = []
        
        def get_best_path(node: TreeNode) -> List[torch.Tensor]:
            path = [node.state]
            if node.children:
                best_child = max(node.children, key=lambda x: x.score)
                path.extend(get_best_path(best_child))
            return path
        
        for root in roots:
            paths.append(get_best_path(root))
            
        return paths
    
    def _refine_states(self, paths: List[List[torch.Tensor]], 
                      original_states: torch.Tensor) -> torch.Tensor:
        """Refine states using reasoning paths."""
        refined_states = []
        
        for path, orig_state in zip(paths, original_states):
            # Combine all states in the path
            path_tensor = torch.stack(path)
            
            # Get final refined state
            path_encoding = path_tensor.mean(dim=0)
            refined = self.state_refiner(
                torch.cat([orig_state, path_encoding], dim=-1)
            )
            
            # Interpolate with original state
            refined = (1 - self.config.rewrite_factor) * orig_state + \
                     self.config.rewrite_factor * refined
            
            refined_states.append(refined)
            
        return torch.stack(refined_states)
    
    def save_pretrained(self, save_path: str):
        """Save tree of thoughts components."""
        torch.save({
            'config': self.config,
            'state_dict': self.state_dict()
        }, f"{save_path}/tree_of_thoughts.pt")
        
    @classmethod
    def from_pretrained(cls, load_path: str, model: PreTrainedModel):
        """Load tree of thoughts components."""
        checkpoint = torch.load(f"{load_path}/tree_of_thoughts.pt")
        instance = cls(model=model, config=checkpoint['config'])
        instance.load_state_dict(checkpoint['state_dict'])
        return instance
