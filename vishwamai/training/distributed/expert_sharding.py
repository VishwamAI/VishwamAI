"""Expert sharding strategies for distributed MoE on TPUs."""

from typing import Optional, Dict, List, Tuple, Any
import math

import torch
import torch.distributed as dist
import torch_xla.core.xla_model as xm
import torch_xla.distributed.xla_multiprocessing as xmp

from ...model.moe.expert import ExpertNetwork
from ...utils.logging import get_logger

logger = get_logger(__name__)

class ExpertShardingManager:
    """Manager for sharding experts across TPU cores."""
    
    def __init__(
        self,
        num_experts: int,
        num_tpu_cores: int,
        strategy: str = "axis_0",
        local_dispatch: bool = True,
        balance_load: bool = True,
        device: Optional[torch.device] = None,
    ):
        """Initialize expert sharding manager.
        
        Args:
            num_experts: Total number of experts
            num_tpu_cores: Number of TPU cores
            strategy: Sharding strategy ("axis_0", "axis_1", "replicated")
            local_dispatch: Whether to use local expert dispatch
            balance_load: Whether to balance expert load
            device: TPU device
        """
        self.num_experts = num_experts
        self.num_tpu_cores = num_tpu_cores
        self.strategy = strategy
        self.local_dispatch = local_dispatch
        self.balance_load = balance_load
        self.device = device or xm.xla_device()
        
        self.rank = xm.get_ordinal()
        self.world_size = xm.xrt_world_size()
        
        # Compute expert assignments
        self.expert_map = self._create_expert_map()
        self.local_experts = self._get_local_experts()
        
        if self.rank == 0:
            logger.info(
                f"Expert Sharding: {num_experts} experts on {num_tpu_cores} cores"
            )
            logger.info(f"Strategy: {strategy}")
            logger.info(f"Local dispatch: {local_dispatch}")
            
    def _create_expert_map(self) -> Dict[int, List[int]]:
        """Create mapping of TPU cores to expert indices.
        
        Returns:
            Dictionary mapping core index to list of expert indices
        """
        expert_map = {i: [] for i in range(self.num_tpu_cores)}
        
        if self.strategy == "replicated":
            # All experts on all cores
            for core in range(self.num_tpu_cores):
                expert_map[core] = list(range(self.num_experts))
                
        elif self.strategy == "axis_0":
            # Split experts across first dimension
            experts_per_core = math.ceil(self.num_experts / self.num_tpu_cores)
            for expert_idx in range(self.num_experts):
                core_idx = expert_idx // experts_per_core
                if core_idx < self.num_tpu_cores:
                    expert_map[core_idx].append(expert_idx)
                    
        elif self.strategy == "axis_1":
            # Split experts across second dimension
            for expert_idx in range(self.num_experts):
                core_idx = expert_idx % self.num_tpu_cores
                expert_map[core_idx].append(expert_idx)
                
        else:
            raise ValueError(f"Unknown sharding strategy: {self.strategy}")
            
        return expert_map
        
    def _get_local_experts(self) -> List[int]:
        """Get expert indices assigned to current TPU core.
        
        Returns:
            List of expert indices
        """
        return self.expert_map[self.rank]
        
    def shard_experts(
        self,
        experts: List[ExpertNetwork]
    ) -> Tuple[List[ExpertNetwork], Dict[str, Any]]:
        """Shard experts across TPU cores.
        
        Args:
            experts: List of expert networks
            
        Returns:
            Tuple containing:
            - List of experts for current core
            - Sharding metadata dictionary
        """
        if len(experts) != self.num_experts:
            raise ValueError(
                f"Number of experts ({len(experts)}) does not match "
                f"expected number ({self.num_experts})"
            )
            
        # Get experts for this core
        local_experts = [
            experts[idx].to(self.device) 
            for idx in self.local_experts
        ]
        
        # Create metadata
        metadata = {
            "expert_map": self.expert_map,
            "local_experts": self.local_experts,
            "strategy": self.strategy
        }
        
        return local_experts, metadata
        
    def get_dispatch_mask(
        self,
        router_probs: torch.Tensor,
        k: int
    ) -> torch.Tensor:
        """Get expert dispatch mask for local experts.
        
        Args:
            router_probs: Routing probabilities [batch_size, seq_len, num_experts]
            k: Number of experts to select per token
            
        Returns:
            Boolean mask for expert selection
        """
        batch_size, seq_len, _ = router_probs.shape
        
        if self.local_dispatch:
            # Only select from local experts
            mask = torch.zeros_like(router_probs, dtype=torch.bool)
            mask[:, :, self.local_experts] = True
            
            # Ensure at least k experts available
            if len(self.local_experts) < k:
                raise ValueError(
                    f"Not enough local experts ({len(self.local_experts)}) "
                    f"for k={k}"
                )
        else:
            # Allow selection of any expert
            mask = torch.ones_like(router_probs, dtype=torch.bool)
            
        return mask
        
    def all_gather_expert_outputs(
        self,
        local_outputs: torch.Tensor,
        expert_counts: torch.Tensor
    ) -> torch.Tensor:
        """Gather expert outputs from all TPU cores.
        
        Args:
            local_outputs: Output tensor from local experts
            expert_counts: Number of tokens per expert
            
        Returns:
            Gathered outputs from all cores
        """
        if self.strategy == "replicated":
            # No need to gather for replicated experts
            return local_outputs
            
        # Gather outputs from all cores
        gathered_outputs = [
            torch.zeros_like(local_outputs) 
            for _ in range(self.world_size)
        ]
        dist.all_gather(gathered_outputs, local_outputs)
        
        # Sum outputs (each expert appears once)
        outputs = torch.zeros_like(local_outputs)
        for core_idx, core_outputs in enumerate(gathered_outputs):
            core_experts = self.expert_map[core_idx]
            outputs[:, :, core_experts] = core_outputs[:, :, core_experts]
            
        return outputs
        
    def balance_expert_load(
        self,
        router_probs: torch.Tensor,
        expert_counts: torch.Tensor
    ) -> torch.Tensor:
        """Balance load across experts.
        
        Args:
            router_probs: Routing probabilities
            expert_counts: Number of tokens per expert
            
        Returns:
            Adjusted routing probabilities
        """
        if not self.balance_load:
            return router_probs
            
        # Compute load balancing loss
        # P_i = fraction of tokens routed to expert i
        # L_aux = \sum_i (P_i * num_experts - 1)^2
        probs_per_expert = router_probs.mean(dim=[0, 1])  # [num_experts]
        aux_loss = torch.mean(
            (probs_per_expert * self.num_experts - 1.0) ** 2
        )
        
        # Adjust routing probabilities
        if self.local_dispatch:
            # Only adjust local experts
            router_probs[:, :, self.local_experts] *= torch.exp(-aux_loss)
        else:
            # Adjust all experts
            router_probs *= torch.exp(-aux_loss)
            
        # Renormalize
        router_probs /= router_probs.sum(dim=-1, keepdim=True)
        
        return router_probs
        
    def expert_capacity(self, batch_size: int, seq_len: int) -> int:
        """Compute expert capacity (max tokens per expert).
        
        Args:
            batch_size: Batch size
            seq_len: Sequence length
            
        Returns:
            Maximum number of tokens per expert
        """
        tokens_per_core = batch_size * seq_len
        num_local_experts = len(self.local_experts)
        
        if num_local_experts == 0:
            return 0
            
        # Capacity = tokens_per_core / num_local_experts * capacity_factor
        capacity = int(
            tokens_per_core / num_local_experts * 1.25  # 1.25x capacity factor
        )
        
        return capacity
