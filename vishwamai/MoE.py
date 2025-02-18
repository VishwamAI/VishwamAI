import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F
from typing import Optional, Tuple
from .config import ModelArgs
from .parallel import ColumnParallelLinear, RowParallelLinear

# Default values for distributed training
world_size = dist.get_world_size() if dist.is_initialized() else 1
rank = dist.get_rank() if dist.is_initialized() else 0

def create_local_experts(base_expert: nn.Module, num_experts: int) -> nn.ModuleList:
    """Create local experts by deepcopying base expert."""
    return nn.ModuleList([base_expert for _ in range(num_experts)])

class MoELayer(nn.Module):
    """Mixture of Experts Layer."""
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.dim = args.dim
        self.moe_inter_dim = args.moe_inter_dim
        
        # Ensure number of experts is valid
        assert args.n_routed_experts % world_size == 0
        self.num_local_experts = args.n_routed_experts // world_size
        
        # Create experts
        self.w1 = ColumnParallelLinear(self.dim, self.moe_inter_dim, bias=False)
        self.w2 = RowParallelLinear(self.moe_inter_dim, self.dim, bias=False)
        self.w3 = ColumnParallelLinear(self.dim, self.moe_inter_dim, bias=False)
        
        # Create expert routing
        self.router = nn.Linear(self.dim, args.n_routed_experts, bias=False)
        self.gate = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with expert routing."""
        batch_size, seq_len, hidden_dim = x.shape
        x_reshaped = x.view(-1, hidden_dim)
        
        # Get routing scores
        route_logits = self.router(x_reshaped)
        route_probs = self.gate(route_logits)
        
        # Select top-k experts
        top_k = min(self.args.n_activated_experts, self.num_local_experts)
        routes_k, indices_k = torch.topk(route_probs, top_k, dim=-1)
        
        # Normalize selected probabilities
        routes_k_norm = routes_k / routes_k.sum(dim=-1, keepdim=True)
        
        # Expert computation
        expert_outputs = torch.zeros_like(x_reshaped)
        for idx, gate in zip(indices_k.t(), routes_k_norm.t()):
            h = F.silu(self.w1(x_reshaped))
            h = self.w3(h)
            h = F.silu(h)
            h = self.w2(h)
            expert_outputs.scatter_add_(0, idx.unsqueeze(1).expand(-1, hidden_dim), gate.unsqueeze(1) * h)
        
        return expert_outputs.view(batch_size, seq_len, hidden_dim)

class MoE(nn.Module):
    """Mixture of Experts wrapper for transformer."""
    def __init__(self, model: nn.Module, args: Optional[ModelArgs] = None):
        super().__init__()
        # Save the base model
        self.base_model = model
        
        # Get config from either args or base model
        if args is None:
            if hasattr(model, 'args'):
                args = model.args
            else:
                raise ValueError("Either args must be provided or base model must have args attribute")
        
        # Store configuration
        self.args = args
        self.dim = args.dim
        
        # Create MoE layer
        self.moe = MoELayer(args)
        
    def forward(self, *args, **kwargs):
        """Forward pass through base model and MoE layer."""
        # Get base model output
        base_output = self.base_model(*args, **kwargs)
        
        # Apply MoE layer
        if isinstance(base_output, tuple):
            output = self.moe(base_output[0])
            return (output,) + base_output[1:]
        else:
            return self.moe(base_output)
