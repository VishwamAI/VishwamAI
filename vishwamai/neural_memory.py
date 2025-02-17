import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import math

class NeuralMemory(nn.Module):
    """
    Neural memory module implementing hierarchical memory with short-term and long-term storage.
    
    This module provides a trainable memory mechanism that can store and retrieve information
    with different temporal scales, implementing sparse access patterns and importance-based
    memory management.
    """
    
    def __init__(
        self,
        memory_size: int,
        hidden_dim: int = 768,
        num_heads: int = 8,
        sparsity: float = 0.9
    ):
        """
        Initialize the neural memory module.
        
        Args:
            memory_size: Total size of memory
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            sparsity: Sparsity factor for attention (0 to 1)
            
        Raises:
            ValueError: If parameters are invalid
        """
        super().__init__()
        
        if memory_size < 4:
            raise ValueError("Memory size must be at least 4")
        if hidden_dim % num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if not 0 <= sparsity <= 1:
            raise ValueError("sparsity must be between 0 and 1")
        
        self.memory_size = memory_size
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.sparsity = sparsity
        
        # Initialize hierarchical memory structure
        self._init_memory_structure()
        
        # Initialize network components
        self._init_networks()
        
        # Initialize parameters
        self._init_parameters()
        
        # Track memory usage statistics
        self.access_counts = torch.zeros(memory_size)
        self.last_access_time = torch.zeros(memory_size)
        self.update_step = 0
        
    def _init_memory_structure(self):
        """Initialize hierarchical memory structure."""
        # Hierarchical memory components
        self.short_term_size = max(4, self.memory_size // 4)
        self.long_term_size = self.memory_size - self.short_term_size
        
        # Memory components with hierarchical structure
        self.short_term_keys = nn.Parameter(torch.randn(self.short_term_size, self.hidden_dim))
        self.short_term_values = nn.Parameter(torch.randn(self.short_term_size, self.hidden_dim))
        self.long_term_keys = nn.Parameter(torch.randn(self.long_term_size, self.hidden_dim))
        self.long_term_values = nn.Parameter(torch.randn(self.long_term_size, self.hidden_dim))
        
        # Memory importance scores
        self.short_term_importance = nn.Parameter(torch.ones(self.short_term_size))
        self.long_term_importance = nn.Parameter(torch.ones(self.long_term_size))
        
    def _init_networks(self):
        """Initialize neural network components."""
        # Memory update components
        self.key_transform = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.value_transform = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.query_transform = nn.Linear(self.hidden_dim, self.hidden_dim)
        
        # Memory control gates
        self.write_gate = nn.Linear(self.hidden_dim * 2, 1)
        self.read_gate = nn.Linear(self.hidden_dim * 2, 1)
        
        # Memory optimization components
        self.compression_layer = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.expansion_layer = nn.Linear(self.hidden_dim // 2, self.hidden_dim)
        
        self.layer_norm = nn.LayerNorm(self.hidden_dim)
        
    def _init_parameters(self):
        """Initialize model parameters."""
        # Initialize memory components
        for param in [self.short_term_keys, self.short_term_values,
                     self.long_term_keys, self.long_term_values]:
            nn.init.normal_(param, mean=0.0, std=0.02)
        
        # Initialize transformation layers
        for module in [self.key_transform, self.value_transform, self.query_transform]:
            nn.init.normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)
            
        # Initialize gates
        for gate in [self.write_gate, self.read_gate]:
            nn.init.normal_(gate.weight, std=0.02)
            nn.init.zeros_(gate.bias)
            
        # Initialize compression components
        nn.init.xavier_uniform_(self.compression_layer.weight)
        nn.init.zeros_(self.compression_layer.bias)
        nn.init.xavier_uniform_(self.expansion_layer.weight)
        nn.init.zeros_(self.expansion_layer.bias)
    
    def forward(
        self,
        inputs: torch.Tensor,
        update_memory: bool = True
    ) -> Tuple[torch.Tensor, Optional[Dict[str, torch.Tensor]]]:
        """
        Process inputs through neural memory.
        
        Args:
            inputs: Input tensor of shape (batch_size, seq_len, hidden_dim)
            update_memory: Whether to update memory contents
            
        Returns:
            Tuple of (output tensor, optional attention info dictionary)
            
        Raises:
            RuntimeError: If input tensor has incorrect shape
        """
        if inputs.dim() != 3 or inputs.size(-1) != self.hidden_dim:
            raise RuntimeError(
                f"Expected input shape (batch_size, seq_len, {self.hidden_dim}), "
                f"got {inputs.shape}"
            )
            
        try:
            batch_size = inputs.size(0)
            
            # Transform inputs for multi-head attention
            queries = self._split_heads(self.query_transform(inputs))
            keys = self._split_heads(self.key_transform(inputs))
            values = self._split_heads(self.value_transform(inputs))
            
            # Process memories with importance-weighted attention
            short_term_output, st_attention = self._attend_memory(
                queries,
                self._split_heads(self.short_term_keys.unsqueeze(0)),
                self._split_heads(self.short_term_values.unsqueeze(0)),
                self.short_term_importance,
                is_short_term=True
            )
            
            long_term_output, lt_attention = self._attend_memory(
                queries,
                self._split_heads(self.long_term_keys.unsqueeze(0)),
                self._split_heads(self.long_term_values.unsqueeze(0)),
                self.long_term_importance,
                is_short_term=False
            )
            
            # Combine outputs with adaptive gating
            combined_output = self._combine_memory_outputs(
                inputs,
                self._combine_heads(short_term_output),
                self._combine_heads(long_term_output)
            )
            
            # Update memory if required
            if update_memory and self.training:
                self._update_memory(
                    self._combine_heads(keys),
                    self._combine_heads(values),
                    st_attention,
                    lt_attention
                )
                
            # Update statistics
            self._update_statistics(st_attention, lt_attention)
            
            return combined_output, {
                'short_term_attention': st_attention,
                'long_term_attention': lt_attention
            }
            
        except Exception as e:
            raise RuntimeError(f"Error in neural memory forward pass: {str(e)}") from e
    
    def _attend_memory(
        self,
        queries: torch.Tensor,
        keys: torch.Tensor,
        values: torch.Tensor,
        importance: torch.Tensor,
        is_short_term: bool
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform attention with sparse access and importance weighting.
        
        Args:
            queries: Query tensor
            keys: Key tensor
            values: Value tensor
            importance: Importance scores
            is_short_term: Whether this is short-term memory
            
        Returns:
            Tuple of (output tensor, attention weights)
        """
        # Calculate attention scores
        scores = torch.matmul(queries, keys.transpose(-2, -1))
        scores = scores / math.sqrt(self.head_dim)
        
        # Apply importance weighting with temporal decay
        temporal_decay = 0.99 if is_short_term else 0.999
        decay_factor = temporal_decay ** self.update_step
        scores = scores * (importance.view(1, 1, 1, -1) * decay_factor)
        
        # Implement sparse attention during training
        if self.training:
            k = int((1 - self.sparsity) * scores.size(-1))
            topk_scores, _ = torch.topk(scores, k, dim=-1)
            threshold = topk_scores[..., -1, None]
            scores = torch.where(
                scores >= threshold,
                scores,
                torch.full_like(scores, float('-inf'))
            )
        
        # Apply sliding window attention during inference
        else:
            window_size = min(128, scores.size(-1))
            padding = window_size // 2
            padded_scores = F.pad(scores, (padding, padding), value=float('-inf'))
            scores = padded_scores.unfold(-1, window_size, 1)
            scores = scores[..., :scores.size(-2)]
        
        attention = F.softmax(scores, dim=-1)
        output = torch.matmul(attention, values)
        
        return output, attention

    def _combine_memory_outputs(
        self,
        inputs: torch.Tensor,
        short_term: torch.Tensor,
        long_term: torch.Tensor
    ) -> torch.Tensor:
        """Combine memory outputs with adaptive gating."""
        # Calculate importance-based weights
        st_weight = torch.sigmoid(self.short_term_importance.mean())
        lt_weight = torch.sigmoid(self.long_term_importance.mean())
        
        # Normalize weights
        total = st_weight + lt_weight
        st_weight = st_weight / total
        lt_weight = lt_weight / total
        
        # Combine outputs
        memory_output = st_weight * short_term + lt_weight * long_term
        
        # Apply read gate
        read_gate_input = torch.cat([inputs, memory_output], dim=-1)
        read_gate = torch.sigmoid(self.read_gate(read_gate_input))
        gated_output = memory_output * read_gate
        
        # Residual connection and normalization
        return self.layer_norm(inputs + gated_output)

    def _update_memory(
        self,
        keys: torch.Tensor,
        values: torch.Tensor,
        st_attention: torch.Tensor,
        lt_attention: torch.Tensor
    ):
        """Update hierarchical memory with importance-based management."""
        try:
            # Calculate importance scores for new information
            new_importance = self.calculate_importance(keys, values)
            
            # Compress memory contents for efficiency
            compressed_keys = self.compression_layer(keys)
            compressed_values = self.compression_layer(values)
            
            # Update short-term memory
            self._update_component(
                self.short_term_keys,
                self.short_term_values,
                self.short_term_importance,
                keys[:self.short_term_size],
                values[:self.short_term_size],
                new_importance[:self.short_term_size]
            )
            
            # Selectively transfer to long-term memory
            self._transfer_to_long_term()
            
            # Optimize memory usage
            self._optimize_memory()
            
            # Prune least important memories
            self._prune_memories()
            
        except Exception as e:
            print(f"Warning: Memory update failed: {str(e)}")
    
    def _update_component(
        self,
        comp_keys: torch.Tensor,
        comp_values: torch.Tensor,
        comp_importance: torch.Tensor,
        new_keys: torch.Tensor,
        new_values: torch.Tensor,
        new_importance: torch.Tensor
    ):
        """Update memory component with importance-based merging."""
        with torch.no_grad():
            # Calculate merge weights
            merge_weights = F.softmax(
                torch.stack([comp_importance, new_importance]), dim=0
            )
            
            # Update memory components
            comp_keys.data = (
                comp_keys * merge_weights[0].unsqueeze(-1) +
                new_keys * merge_weights[1].unsqueeze(-1)
            )
            comp_values.data = (
                comp_values * merge_weights[0].unsqueeze(-1) +
                new_values * merge_weights[1].unsqueeze(-1)
            )
            
            # Update importance scores with decay
            decay = 0.99
            comp_importance.data = torch.maximum(
                comp_importance * decay,
                new_importance
            )
    
    def _transfer_to_long_term(self):
        """Transfer important short-term memories to long-term storage."""
        with torch.no_grad():
            # Find important short-term memories
            importance_threshold = torch.quantile(
                self.short_term_importance,
                0.8
            )
            transfer_mask = self.short_term_importance > importance_threshold
            
            if transfer_mask.any():
                # Find least important long-term memories
                _, replace_indices = torch.topk(
                    self.long_term_importance,
                    k=min(transfer_mask.sum().item(), self.long_term_size),
                    largest=False
                )
                
                # Transfer memories
                self.long_term_keys.data[replace_indices] = (
                    self.short_term_keys[transfer_mask][:len(replace_indices)]
                )
                self.long_term_values.data[replace_indices] = (
                    self.short_term_values[transfer_mask][:len(replace_indices)]
                )
                self.long_term_importance.data[replace_indices] = (
                    self.short_term_importance[transfer_mask][:len(replace_indices)]
                )
    
    def _optimize_memory(self):
        """Optimize memory usage through compression and cleanup."""
        with torch.no_grad():
            # Compress rarely accessed memories
            access_mask = self.access_counts < self.access_counts.mean()
            if access_mask.any():
                # Compress keys and values
                compressed_keys = self.compression_layer(
                    self.long_term_keys[access_mask]
                )
                compressed_values = self.compression_layer(
                    self.long_term_values[access_mask]
                )
                
                # Store compressed versions
                self.long_term_keys.data[access_mask] = (
                    self.expansion_layer(compressed_keys)
                )
                self.long_term_values.data[access_mask] = (
                    self.expansion_layer(compressed_values)
                )
    
    def _prune_memories(self):
        """Prune least important memories based on access patterns."""
        with torch.no_grad():
            # Calculate pruning thresholds
            st_threshold = torch.quantile(self.short_term_importance, 0.3)
            lt_threshold = torch.quantile(self.long_term_importance, 0.1)
            
            # Apply pruning with reinitialization
            self._prune_component(
                self.short_term_importance,
                st_threshold,
                self.short_term_keys,
                self.short_term_values
            )
            self._prune_component(
                self.long_term_importance,
                lt_threshold,
                self.long_term_keys,
                self.long_term_values
            )
    
    def _prune_component(
        self,
        importance: torch.Tensor,
        threshold: float,
        keys: torch.Tensor,
        values: torch.Tensor
    ):
        """Prune specific memory component."""
        prune_mask = importance < threshold
        if prune_mask.any():
            # Reinitialize pruned entries
            keys.data[prune_mask] = torch.randn_like(
                keys[prune_mask]
            ) * 0.02
            values.data[prune_mask] = torch.randn_like(
                values[prune_mask]
            ) * 0.02
            importance.data[prune_mask] = 1.0
    
    def _update_statistics(
        self,
        st_attention: torch.Tensor,
        lt_attention: torch.Tensor
    ):
        """Update memory access statistics."""
        with torch.no_grad():
            # Update access counts
            st_access = (st_attention.mean(dim=(0, 1)) > 0.01).float()
            lt_access = (lt_attention.mean(dim=(0, 1)) > 0.01).float()
            
            self.access_counts[:self.short_term_size] += st_access
            self.access_counts[self.short_term_size:] += lt_access
            
            # Update last access time
            self.last_access_time = torch.where(
                torch.cat([st_access, lt_access]) > 0,
                torch.full_like(self.last_access_time, self.update_step),
                self.last_access_time
            )
            
            self.update_step += 1
    
    def _split_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Split tensor into attention heads."""
        batch_size = tensor.size(0)
        tensor = tensor.view(batch_size, -1, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)
    
    def _combine_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        """Combine attention heads."""
        tensor = tensor.transpose(1, 2)
        batch_size = tensor.size(0)
        seq_length = tensor.size(1)
        return tensor.contiguous().view(batch_size, seq_length, self.hidden_dim)
    
    def calculate_importance(
        self,
        keys: torch.Tensor,
        values: torch.Tensor
    ) -> torch.Tensor:
        """
        Calculate importance scores for new information.
        
        Args:
            keys: Key tensor
            values: Value tensor
            
        Returns:
            Tensor of importance scores
        """
        # Calculate key-value metrics
        key_norm = torch.norm(keys, dim=-1)
        value_norm = torch.norm(values, dim=-1)
        similarity = F.cosine_similarity(keys, values, dim=-1)
        
        # Combine metrics with learned weights
        importance = (key_norm + value_norm) * similarity
        return F.sigmoid(importance)
    
    def reset_memory(self):
        """Reset memory to initial state."""
        with torch.no_grad():
            self._init_parameters()
            self.access_counts.zero_()
            self.last_access_time.zero_()
            self.update_step = 0
    
    def get_memory_state(self) -> Dict[str, Any]:
        """
        Get current memory state and statistics.
        
        Returns:
            Dictionary containing memory state and statistics
        """
        with torch.no_grad():
            memory_state = {
                'short_term': {
                    'keys': self.short_term_keys.detach(),
                    'values': self.short_term_values.detach(),
                    'importance': self.short_term_importance.detach()
                },
                'long_term': {
                    'keys': self.long_term_keys.detach(),
                    'values': self.long_term_values.detach(),
                    'importance': self.long_term_importance.detach()
                },
                'statistics': {
                    'access_counts': self.access_counts.clone(),
                    'last_access': self.last_access_time.clone(),
                    'update_step': self.update_step,
                    'short_term_usage': (self.short_term_importance > 0.5).float().mean().item(),
                    'long_term_usage': (self.long_term_importance > 0.5).float().mean().item()
                }
            }
            return memory_state
    
    def set_memory_state(self, state: Dict[str, Any]):
        """
        Set memory state from saved state.
        
        Args:
            state: Dictionary containing memory state
            
        Raises:
            ValueError: If state is incompatible
        """
        try:
            # Verify state compatibility
            for memory_type in ['short_term', 'long_term']:
                for key in ['keys', 'values', 'importance']:
                    stored = state[memory_type][key]
                    current = getattr(self, f"{memory_type}_{key}")
                    if stored.size() != current.size():
                        raise ValueError(
                            f"Incompatible {memory_type} {key} size: "
                            f"got {stored.size()}, expected {current.size()}"
                        )
            
            # Load state
            with torch.no_grad():
                for memory_type in ['short_term', 'long_term']:
                    for key in ['keys', 'values', 'importance']:
                        param = getattr(self, f"{memory_type}_{key}")
                        param.data.copy_(state[memory_type][key])
                
                # Load statistics if available
                if 'statistics' in state:
                    stats = state['statistics']
                    self.access_counts = stats['access_counts'].to(
                        self.short_term_keys.device
                    )
                    self.last_access_time = stats['last_access'].to(
                        self.short_term_keys.device
                    )
                    self.update_step = stats['update_step']
                    
        except Exception as e:
            raise ValueError(f"Error loading memory state: {str(e)}")
    
    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        """
        Save memory state.
        
        Returns:
            Dictionary containing complete module state
        """
        state_dict = super().state_dict(*args, **kwargs)
        state_dict.update({
            'memory_size': self.memory_size,
            'hidden_dim': self.hidden_dim,
            'num_heads': self.num_heads,
            'sparsity': self.sparsity,
            'short_term_size': self.short_term_size,
            'long_term_size': self.long_term_size,
            'statistics': {
                'access_counts': self.access_counts,
                'last_access_time': self.last_access_time,
                'update_step': self.update_step
            }
        })
        return state_dict
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """
        Load memory state.
        
        Args:
            state_dict: Dictionary containing module state
            
        Raises:
            ValueError: If state dict is incompatible
        """
        # Verify compatibility
        for key in ['memory_size', 'hidden_dim', 'num_heads', 'sparsity']:
            if state_dict[key] != getattr(self, key):
                raise ValueError(
                    f"Incompatible {key}: expected {getattr(self, key)}, "
                    f"got {state_dict[key]}"
                )
        
        # Load statistics if available
        if 'statistics' in state_dict:
            stats = state_dict.pop('statistics')
            self.access_counts = stats['access_counts'].to(
                self.short_term_keys.device
            )
            self.last_access_time = stats['last_access_time'].to(
                self.short_term_keys.device
            )
            self.update_step = stats['update_step']
        
        # Remove config items
        for key in ['memory_size', 'hidden_dim', 'num_heads', 'sparsity',
                   'short_term_size', 'long_term_size']:
            state_dict.pop(key, None)
            
        super().load_state_dict(state_dict)
