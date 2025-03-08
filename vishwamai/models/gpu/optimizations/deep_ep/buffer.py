"""
Buffer management for efficient parallel operations on GPU.
Implements optimized memory allocation and transfer for expert parallelism.
"""

import torch
import torch.distributed as dist
from typing import Optional, Tuple, Any, List
import math

from .utils import get_num_sms, get_best_configs

class Buffer:
    """Manages efficient data transfer and storage for parallel operations"""
    
    _global_num_sms = None
    _warps_per_sm = 4
    
    def __init__(self, group: Optional[Any] = None, hidden_bytes: int = 0, 
                 num_nvl_bytes: int = 0, num_rdma_bytes: int = 0):
        """
        Initialize buffer manager
        
        Args:
            group: Optional process group for distributed operations
            hidden_bytes: Size of hidden state buffer in bytes
            num_nvl_bytes: Size of NVLink buffer in bytes
            num_rdma_bytes: Size of RDMA buffer in bytes
        """
        self.group = group
        self.hidden_bytes = hidden_bytes
        
        # Initialize buffers
        self.hidden_buffer = None
        self.nvlink_buffer = None
        self.rdma_buffer = None
        self._alloc_buffers(hidden_bytes, num_nvl_bytes, num_rdma_bytes)
        
        # Cache for dispatch/combine operations
        self._dispatch_cache = {}
        self._combine_cache = {}
        
        # Get optimal configurations
        if torch.cuda.is_available():
            configs = get_best_configs((1024, 1024))  # Default shape
            self._warps_per_sm = configs["num_warps"]
        
    def _alloc_buffers(self, hidden_bytes: int, num_nvl_bytes: int, num_rdma_bytes: int) -> None:
        """Allocate GPU buffers for data transfer and computation"""
        if torch.cuda.is_available():
            if hidden_bytes > 0:
                self.hidden_buffer = torch.cuda.ByteTensor(hidden_bytes)
            if num_nvl_bytes > 0:
                self.nvlink_buffer = torch.cuda.ByteTensor(num_nvl_bytes)
            if num_rdma_bytes > 0:
                self.rdma_buffer = torch.cuda.ByteTensor(num_rdma_bytes)
                
    def get_dispatch_layout(
        self,
        topk_idx: torch.Tensor,
        num_experts: int,
        previous_event: Optional[torch.cuda.Event] = None,
        async_finish: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.cuda.Event]]:
        """
        Calculate optimal layout for expert dispatch
        
        Args:
            topk_idx: Top-k expert indices for each token
            num_experts: Total number of experts
            previous_event: Optional event to sync with
            async_finish: Whether to finish asynchronously
            
        Returns:
            Tuple containing:
            - num_tokens_per_rank: Number of tokens per rank
            - num_tokens_rdma: Number of tokens for RDMA
            - num_tokens_expert: Number of tokens per expert
            - is_token_in_rank: Mask indicating token rank assignment
            - event: CUDA event for synchronization
        """
        batch_size, seq_len = topk_idx.shape[:2]
        device = topk_idx.device
        
        # Calculate token distribution 
        num_tokens = batch_size * seq_len
        tokens_per_expert = torch.zeros(num_experts, dtype=torch.long, device=device)
        for i in range(num_experts):
            tokens_per_expert[i] = (topk_idx == i).sum()
            
        # Get rank info for distributed case
        if self.group is not None:
            world_size = dist.get_world_size(self.group)
            rank = dist.get_rank(self.group)
        else:
            world_size = 1
            rank = 0
            
        # Distribute tokens across ranks
        num_tokens_per_rank = tokens_per_expert // world_size
        num_tokens_rdma = tokens_per_expert % world_size
        
        # Track which tokens go to which rank
        is_token_in_rank = torch.zeros_like(topk_idx, dtype=torch.bool)
        for i in range(num_experts):
            expert_tokens = (topk_idx == i).nonzero()
            rank_size = num_tokens_per_rank[i]
            start_idx = rank * rank_size
            end_idx = start_idx + rank_size
            if start_idx < expert_tokens.size(0):
                end_idx = min(end_idx, expert_tokens.size(0))
                token_indices = expert_tokens[start_idx:end_idx]
                is_token_in_rank[token_indices[:, 0], token_indices[:, 1]] = True
        
        # Create synchronization event
        event = None
        if async_finish and torch.cuda.is_available():
            event = torch.cuda.Event()
            if previous_event is not None:
                previous_event.synchronize()
            event.record()
            
        return num_tokens_per_rank, num_tokens_rdma, tokens_per_expert, is_token_in_rank, event

    def dispatch(
        self,
        x: torch.Tensor,
        topk_idx: torch.Tensor,
        topk_weights: torch.Tensor,
        num_tokens_per_rank: Optional[torch.Tensor] = None,
        num_tokens_per_rdma_rank: Optional[torch.Tensor] = None,
        is_token_in_rank: Optional[torch.Tensor] = None,
        num_tokens_per_expert: Optional[torch.Tensor] = None,
        previous_event: Optional[torch.cuda.Event] = None,
        async_finish: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int], Any, Optional[torch.cuda.Event]]:
        """
        Dispatch tokens to experts
        
        Args:
            x: Input tensor
            topk_idx: Top-k expert indices
            topk_weights: Expert weights
            num_tokens_per_rank: Number of tokens per rank
            num_tokens_per_rdma_rank: Number of tokens for RDMA per rank
            is_token_in_rank: Token rank assignment mask
            num_tokens_per_expert: Number of tokens per expert
            previous_event: Optional event to sync with
            async_finish: Whether to finish asynchronously
            
        Returns:
            Tuple containing:
            - Dispatched tokens
            - Updated indices
            - Updated weights
            - Expert counts
            - Operation handle
            - Event for synchronization
        """
        if previous_event is not None:
            previous_event.synchronize()
            
        # Get dispatch layout if not provided
        if num_tokens_per_rank is None:
            num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, \
                is_token_in_rank, _ = self.get_dispatch_layout(
                    topk_idx, topk_idx.size(-1), async_finish=False
                )
                
        # Initialize output tensors
        batch_size, seq_len = x.shape[:2]
        device = x.device
        dtype = x.dtype
        
        total_expert_tokens = num_tokens_per_expert.sum().item()
        dispatched_shape = list(x.shape)
        dispatched_shape[0] = total_expert_tokens
        
        dispatched_x = torch.zeros(dispatched_shape, dtype=dtype, device=device)
        dispatched_idx = torch.zeros((total_expert_tokens,), dtype=torch.long, device=device)
        dispatched_weights = torch.zeros((total_expert_tokens,), dtype=dtype, device=device)
        
        # Dispatch tokens to experts
        start_idx = 0
        expert_counts = []
        for expert_idx in range(topk_idx.size(-1)):
            expert_mask = (topk_idx == expert_idx) & is_token_in_rank
            num_expert_tokens = expert_mask.sum().item()
            expert_counts.append(num_expert_tokens)
            
            if num_expert_tokens > 0:
                token_indices = expert_mask.nonzero()
                dispatched_x[start_idx:start_idx + num_expert_tokens] = \
                    x[token_indices[:, 0], token_indices[:, 1]]
                dispatched_idx[start_idx:start_idx + num_expert_tokens] = \
                    token_indices[:, 1]
                dispatched_weights[start_idx:start_idx + num_expert_tokens] = \
                    topk_weights[token_indices[:, 0], token_indices[:, 1], expert_idx]
                    
            start_idx += num_expert_tokens
            
        # Create synchronization event
        event = None
        if async_finish and torch.cuda.is_available():
            event = torch.cuda.Event()
            event.record()
            
        return dispatched_x, dispatched_idx, dispatched_weights, expert_counts, None, event

    def combine(
        self,
        expert_outputs: torch.Tensor,
        handle: Any,
        topk_weights: Optional[torch.Tensor] = None,
        prev_event: Optional[torch.cuda.Event] = None
    ) -> Tuple[torch.Tensor, Optional[torch.cuda.Event]]:
        """
        Combine expert outputs
        
        Args:
            expert_outputs: Tensor of expert outputs
            handle: Operation handle from dispatch
            topk_weights: Optional expert weights
            prev_event: Optional event to sync with
            
        Returns:
            Tuple of (combined output, synchronization event)
        """
        if prev_event is not None:
            prev_event.synchronize()
            
        if handle is not None:
            # Use the handle to get original dimensions and mapping
            pass
            
        # If no handle, assume simple averaging
        if topk_weights is not None:
            combined = expert_outputs * topk_weights.unsqueeze(-1)
        else:
            combined = expert_outputs
            
        # Create synchronization event
        event = None
        if torch.cuda.is_available():
            event = torch.cuda.Event()
            event.record()
            
        return combined, event
