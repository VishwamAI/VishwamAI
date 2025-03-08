"""
Expert Parallelism Load Balancer (EPLB) for efficient MoE computation
"""

import torch
import torch.nn.functional as F

class EPLB:
    """Expert Parallelism Load Balancer"""
    
    def __init__(self, num_experts, capacity_factor=1.2):
        self.num_experts = num_experts
        self.capacity_factor = capacity_factor
        self.expert_counts = None
        self.expert_capacity = None
        
    def calculate_capacity(self, batch_size, seq_len):
        """Calculate expert capacity based on token count"""
        # Total tokens with capacity factor
        total_tokens = batch_size * seq_len
        tokens_per_expert = total_tokens // self.num_experts
        self.expert_capacity = int(tokens_per_expert * self.capacity_factor)
        
    def get_expert_assignment(self, router_logits):
        """Get load-balanced expert assignments"""
        batch_size, seq_len, num_experts = router_logits.shape
        device = router_logits.device
        
        # Calculate expert capacity if not set
        if self.expert_capacity is None:
            self.calculate_capacity(batch_size, seq_len)
            
        # Get top-k experts (k=2)
        top_2_logits, top_2_indices = torch.topk(router_logits, k=2, dim=-1)
        router_probs = F.softmax(top_2_logits, dim=-1)
        
        # Track expert assignment counts
        self.expert_counts = torch.zeros(num_experts, dtype=torch.int32, device=device)
        
        # Initialize assignments
        final_indices = torch.zeros_like(top_2_indices[:,:,0])
        final_weights = torch.zeros_like(router_probs[:,:,0])
        
        # Assign tokens to experts with load balancing
        for i in range(batch_size):
            for j in range(seq_len):
                # Try primary expert first
                primary_expert = top_2_indices[i,j,0]
                if self.expert_counts[primary_expert] < self.expert_capacity:
                    final_indices[i,j] = primary_expert
                    final_weights[i,j] = router_probs[i,j,0]
                    self.expert_counts[primary_expert] += 1
                
                # Try secondary expert if primary is full
                else:
                    secondary_expert = top_2_indices[i,j,1]
                    if self.expert_counts[secondary_expert] < self.expert_capacity:
                        final_indices[i,j] = secondary_expert
                        final_weights[i,j] = router_probs[i,j,1]
                        self.expert_counts[secondary_expert] += 1
                    else:
                        # Both experts full, assign to least loaded
                        least_loaded = self.expert_counts.argmin()
                        final_indices[i,j] = least_loaded
                        final_weights[i,j] = router_probs[i,j, top_2_indices[i,j]==least_loaded][0]
                        self.expert_counts[least_loaded] += 1
                        
        return final_indices, final_weights
        
    def get_load_balancing_loss(self, router_probs):
        """Calculate load balancing auxiliary loss"""
        # Expert assignment fraction across batch
        expert_fraction = router_probs.mean(dim=[0,1])
        target_fraction = torch.ones_like(expert_fraction) / self.num_experts
        
        # Expert load balance loss
        load_balance_loss = torch.sum((expert_fraction - target_fraction).pow(2))
        
        # Expert overflow loss if counts available
        if self.expert_counts is not None:
            overflow_fraction = torch.clamp(
                self.expert_counts.float() / self.expert_capacity - 1.0,
                min=0.0
            ).mean()
            
            load_balance_loss = load_balance_loss + overflow_fraction
            
        return load_balance_loss
        
    def reset_counts(self):
        """Reset expert counts between forward passes"""
        self.expert_counts = None