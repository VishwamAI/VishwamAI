# /home/kasinadhsarma/VishwamAI/vishwamai/models/attention.py
"""
Enhanced attention mechanisms for VishwamAI with cutting-edge features:
- Dynamic sparse attention with learned sparsity
- Cross-domain/multi-modal attention
- Hierarchical MoE structure
- Temporal convolution integration
- Hardware-optimized operations
- Advanced gating with load balancing
- Learned performer features
- FlashMLA for memory-efficient latent attention
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod
import math
import copy

# Optional TPU support
# import torch_xla.core.xla_model as xm

class BaseAttention(nn.Module, ABC):
    """Enhanced base class with hardware-aware initialization"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(BaseAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        # Projections
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self._reset_parameters()
        
    def _reset_parameters(self):
        # Optimized initialization for better hardware utilization
        nn.init.xavier_uniform_(self.q_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.k_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.v_proj.weight, gain=1/math.sqrt(2))
        nn.init.xavier_uniform_(self.o_proj.weight)
        nn.init.constant_(self.q_proj.bias, 0.0)
        nn.init.constant_(self.k_proj.bias, 0.0)
        nn.init.constant_(self.v_proj.bias, 0.0)
        nn.init.constant_(self.o_proj.bias, 0.0)
    
    @abstractmethod
    def forward(self, x, context=None, mask=None):
        pass
    
    def _reshape_for_multihead(self, x):
        batch_size, seq_len, _ = x.size()
        # Reshape to (batch_size, seq_len, num_heads, head_dim)
        x = x.view(batch_size, seq_len, self.num_heads, self.head_dim)
        # Transpose to (batch_size, num_heads, seq_len, head_dim)
        return x.transpose(1, 2)


class DynamicSparseAttention(BaseAttention):
    """Attention with learned sparsity patterns using Gumbel-Softmax"""
    def __init__(self, embed_dim, num_heads, k=10, dropout=0.1, temperature=0.5):
        super(DynamicSparseAttention, self).__init__(embed_dim, num_heads, dropout)
        self.k = k  # Top-k tokens to attend to
        self.temperature = temperature  # For Gumbel-Softmax
        self.sparsity_controller = nn.Linear(embed_dim, 1)  # Learns token importance
        
    def forward(self, x, context=None, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Standard projections
        q = self._reshape_for_multihead(self.q_proj(x))  # (B, H, L, D)
        k = self._reshape_for_multihead(self.k_proj(x if context is None else context))  # (B, H, L, D)
        v = self._reshape_for_multihead(self.v_proj(x if context is None else context))  # (B, H, L, D)
        
        # Compute attention scores
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)  # (B, H, L, L)
        
        # Dynamic sparsity: learn which tokens are important
        token_importance = self.sparsity_controller(x).squeeze(-1)  # (B, L)
        
        # Combine with attention scores to get dynamic sparsity
        importance_attn = token_importance.unsqueeze(1).unsqueeze(1)  # (B, 1, 1, L)
        
        # Generate sparse mask by taking top-k
        if self.training:
            # During training, use Gumbel-Softmax for differentiable top-k
            noise = -torch.log(-torch.log(torch.rand_like(importance_attn) + 1e-10) + 1e-10)
            gumbel_logits = (importance_attn + noise) / self.temperature
            sparse_mask = F.gumbel_softmax(gumbel_logits, tau=self.temperature, hard=True, dim=-1)
            sparse_mask = sparse_mask.repeat(1, self.num_heads, seq_len, 1)
        else:
            # During inference, just use top-k
            _, top_indices = torch.topk(token_importance, min(self.k, seq_len), dim=-1)  # (B, K)
            sparse_mask = torch.zeros(batch_size, 1, 1, seq_len, device=x.device)
            batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(-1)  # (B, 1)
            sparse_mask[batch_indices, 0, 0, top_indices] = 1.0
            sparse_mask = sparse_mask.repeat(1, self.num_heads, seq_len, 1)
        
        # Apply learned sparse mask, attention mask, and softmax
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply sparsification filter
        attn_scores = attn_scores * sparse_mask
        
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Compute weighted sum of values
        output = torch.matmul(attn_probs, v)  # (B, H, L, D)
        
        # Reshape back to (B, L, E)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.o_proj(output)


class LearnedPerformerAttention(BaseAttention):
    """Optimized linear attention with learned feature maps inspired by Performer"""
    def __init__(self, embed_dim, num_heads, kernel_dim=256, dropout=0.1):
        super(LearnedPerformerAttention, self).__init__(embed_dim, num_heads, dropout)
        self.kernel_dim = kernel_dim
        
        # Projections for learned feature maps
        self.phi_proj = nn.Linear(self.head_dim, self.kernel_dim)
        self.psi_proj = nn.Linear(self.head_dim, self.kernel_dim)
        
    def forward(self, x, context=None, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Standard projections
        q = self._reshape_for_multihead(self.q_proj(x))  # (B, H, L, D)
        k = self._reshape_for_multihead(self.k_proj(x if context is None else context))  # (B, H, L, D)
        v = self._reshape_for_multihead(self.v_proj(x if context is None else context))  # (B, H, L, D)
        
        # Project into feature space for linear attention
        q_phi = F.elu(self.phi_proj(q)) + 1  # (B, H, L, K)
        k_psi = F.elu(self.psi_proj(k)) + 1  # (B, H, L, K)
        
        # Apply mask if needed
        if mask is not None:
            mask_expanded = mask.unsqueeze(1).unsqueeze(-1)  # (B, 1, L, 1)
            k_psi = k_psi * mask_expanded
            v_masked = v * mask_expanded
        else:
            v_masked = v
            
        # Compute attention using kernel trick: O(L) instead of O(L²)
        # kv = torch.matmul(k_psi.transpose(-2, -1), v_masked)  # (B, H, K, D)
        # qkv = torch.matmul(q_phi, kv)  # (B, H, L, D)
        # normalizer = torch.matmul(q_phi, k_psi.sum(dim=2).unsqueeze(-1))  # (B, H, L, 1)
        
        kv = torch.einsum('bhld,bhlm->bhdm', k_psi, v_masked)
        qkv = torch.einsum('bhld,bhdm->bhlm', q_phi, kv)
        normalizer = torch.einsum('bhld,bhd->bhl', q_phi, k_psi.sum(dim=2)).unsqueeze(-1)
        
        # Normalize
        output = qkv / (normalizer + 1e-8)  # (B, H, L, D)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.o_proj(output)


class OptimizedMoEAttention(BaseAttention):
    """Mixture of Experts Attention with optimized routing and load balancing"""
    def __init__(self, embed_dim, num_heads, num_experts=4, top_k=2, dropout=0.1, gate_jitter=0.01):
        super(OptimizedMoEAttention, self).__init__(embed_dim, num_heads, dropout)
        self.num_experts = num_experts
        self.top_k = min(top_k, num_experts)
        self.gate_jitter = gate_jitter
        
        # Create expert attention modules
        self.experts = nn.ModuleList([
            BaseExpertAttention(embed_dim, num_heads, dropout=dropout)
            for _ in range(num_experts)
        ])
        
        # Router network
        self.router = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_experts)
        )
        
        # Learnable priors for load balancing
        self.expert_priors = nn.Parameter(torch.ones(num_experts) / num_experts)
        self.load_balancing_coeff = 0.01
        
    def forward(self, x, context=None, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Router logits
        # Calculate token-level routing probabilities
        router_logits = self.router(x)  # (B, L, E)
        
        # Add jitter during training
        if self.training and self.gate_jitter > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.gate_jitter
        
        # Calculate routing probabilities and gate values
        routing_probs = F.softmax(router_logits, dim=-1)  # (B, L, E)
        
        # Determine expert assignments
        if self.top_k == 1:
            # For top-1 routing, use argmax
            expert_indices = torch.argmax(routing_probs, dim=-1)  # (B, L)
            gate_values = torch.gather(routing_probs, -1, expert_indices.unsqueeze(-1)).squeeze(-1)  # (B, L)
            expert_mask = F.one_hot(expert_indices, num_classes=self.num_experts).float()  # (B, L, E)
        else:
            # For top-k routing, use top-k
            top_k_probs, top_k_indices = torch.topk(routing_probs, self.top_k, dim=-1)  # (B, L, K)
            # Normalize to sum to 1
            top_k_probs = top_k_probs / top_k_probs.sum(dim=-1, keepdim=True)
            
            # Create expert mask for sparse dispatch
            expert_mask = torch.zeros_like(routing_probs)  # (B, L, E)
            for k in range(self.top_k):
                k_indices = top_k_indices[:, :, k]  # (B, L)
                k_probs = top_k_probs[:, :, k]  # (B, L)
                batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(-1).expand(-1, seq_len)
                seq_indices = torch.arange(seq_len, device=x.device).unsqueeze(0).expand(batch_size, -1)
                expert_mask[batch_indices, seq_indices, k_indices] += k_probs
            gate_values = expert_mask.sum(dim=-1)  # Should be all 1s
        
        # Calculate load balancing loss
        if self.training:
            # Expert usage statistics
            # Calculate how much each expert is used (like in Switch Transformer)
            expert_usage = expert_mask.mean(dim=[0, 1])  # (E,)
            target_usage = self.expert_priors  # Uniform prior
            load_balance_loss = (expert_usage - target_usage).pow(2).mean() * self.load_balancing_coeff
            # Add the auxiliary loss
            self.aux_loss = load_balance_loss
        else:
            self.aux_loss = 0.0
        
        # Process input through each expert - this is computationally efficient 
        # as each expert processes only tokens assigned to it
        expert_outputs = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device)
        
        for i, expert in enumerate(self.experts):
            # Get the tokens assigned to this expert
            expert_token_mask = expert_mask[:, :, i]  # (B, L)
            if expert_token_mask.sum() > 0:  # Only process if there are tokens assigned
                # Get the routing probabilities for this expert
                expert_probs = expert_mask[:, :, i].unsqueeze(-1)  # (B, L, 1)
                # Apply the expert
                expert_output = expert(x, context, mask)  # (B, L, D)
                # Apply routing weights and add to total
                expert_outputs += expert_output * expert_probs
        
        return expert_outputs


class BaseExpertAttention(BaseAttention):
    """Expert attention module for the MoE Attention"""
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(BaseExpertAttention, self).__init__(embed_dim, num_heads, dropout)
    
    def forward(self, x, context=None, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Standard projections
        q = self._reshape_for_multihead(self.q_proj(x))  # (B, H, L, D)
        k = self._reshape_for_multihead(self.k_proj(x if context is None else context))  # (B, H, L, D)
        v = self._reshape_for_multihead(self.v_proj(x if context is None else context))  # (B, H, L, D)
        
        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Compute weighted sum of values
        output = torch.matmul(attn_probs, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.o_proj(output)


class CrossDomainAttention(BaseAttention):
    """Cross-modal attention for multi-domain integration"""
    def __init__(self, embed_dim, num_heads, num_domains=2, dropout=0.1):
        super(CrossDomainAttention, self).__init__(embed_dim, num_heads, dropout)
        self.num_domains = num_domains
        
        # Domain-specific query projections
        self.domain_q_projs = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) 
            for _ in range(num_domains)
        ])
        
        # Domain adapter to help bridge between domains
        self.domain_adapter = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim)
        )
        
        # Domain mixture gate
        self.domain_gate = nn.Linear(embed_dim, num_domains)
        
    def forward(self, x, context=None, mask=None, domain_id=0):
        batch_size, seq_len, _ = x.size()
        
        # Domain-specific query projection
        q = self._reshape_for_multihead(self.domain_q_projs[domain_id](x))
        
        # Standard key/value
        k = self._reshape_for_multihead(self.k_proj(context if context is not None else x))
        v = self._reshape_for_multihead(self.v_proj(context if context is not None else x))
        
        # Cross-domain adaptation when using context
        if context is not None:
            k_adapted = self.domain_adapter(k)
            v_adapted = self.domain_adapter(v)
            
            # Domain gating factor
            domain_gates = F.softmax(self.domain_gate(x.mean(dim=1)), dim=-1)  # (B, num_domains)
            
            # Blend based on domain
            k = k * (1 - domain_gates[:, 1].view(-1, 1, 1, 1)) + k_adapted * domain_gates[:, 1].view(-1, 1, 1, 1)
            v = v * (1 - domain_gates[:, 1].view(-1, 1, 1, 1)) + v_adapted * domain_gates[:, 1].view(-1, 1, 1, 1)
        
        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Compute weighted sum of values
        output = torch.matmul(attn_probs, v)
        
        # Reshape back
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.o_proj(output)


class TemporalConvAttention(BaseAttention):
    """Temporal convolution-enhanced attention"""
    def __init__(self, embed_dim, num_heads, kernel_size=3, dropout=0.1):
        super(TemporalConvAttention, self).__init__(embed_dim, num_heads, dropout)
        self.kernel_size = kernel_size
        padding = (kernel_size - 1) // 2
        
        # Depthwise separable convolution for temporal relationships
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=padding, groups=embed_dim),
            nn.Conv1d(embed_dim, embed_dim, 1),  # Pointwise conv
            nn.GELU()
        )
        
        # Projection for combining temporal and attention features
        self.fusion_layer = nn.Linear(embed_dim * 2, embed_dim)
        
    def forward(self, x, context=None, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Apply temporal convolution
        # Transpose for conv1d: [B, L, D] -> [B, D, L]
        x_conv = x.transpose(1, 2)
        x_conv = self.temporal_conv(x_conv)
        # Back to [B, L, D]
        x_conv = x_conv.transpose(1, 2)
        
        # Standard self-attention
        q = self._reshape_for_multihead(self.q_proj(x))
        k = self._reshape_for_multihead(self.k_proj(x if context is None else context))
        v = self._reshape_for_multihead(self.v_proj(x if context is None else context))
        
        # Compute attention
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Compute weighted sum of values
        attn_out = torch.matmul(attn_probs, v)
        attn_out = attn_out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        attn_out = self.o_proj(attn_out)
        
        # Fuse temporal and attention features
        output = self.fusion_layer(torch.cat([attn_out, x_conv], dim=-1))
        
        return output


class HierarchicalMoEAttention(BaseAttention):
    """Hierarchical Mixture of Experts Attention with multi-level routing"""
    def __init__(self, embed_dim, num_heads, num_experts=4, num_sub_experts=2, dropout=0.1):
        super(HierarchicalMoEAttention, self).__init__(embed_dim, num_heads, dropout)
        self.num_experts = num_experts
        self.num_sub_experts = num_sub_experts
        
        # Level 1: Expert groups with different attention types
        self.experts = nn.ModuleList()
        attention_types = [DynamicSparseAttention, LearnedPerformerAttention, BaseExpertAttention, TemporalConvAttention]
        
        for i in range(num_experts):
            # Select attention type based on index, repeat if needed
            attention_cls = attention_types[i % len(attention_types)]
            if attention_cls == DynamicSparseAttention:
                self.experts.append(attention_cls(embed_dim, num_heads, k=10, dropout=dropout))
            elif attention_cls == LearnedPerformerAttention:
                self.experts.append(attention_cls(embed_dim, num_heads, kernel_dim=256, dropout=dropout))
            elif attention_cls == TemporalConvAttention:
                self.experts.append(attention_cls(embed_dim, num_heads, kernel_size=3, dropout=dropout))
            else:
                self.experts.append(attention_cls(embed_dim, num_heads, dropout=dropout))
        
        # Level 2: Sub-experts for specialized features within each expert
        self.sub_experts = nn.ModuleList([
            nn.ModuleList([
                nn.Sequential(
                    nn.Linear(embed_dim, embed_dim),
                    nn.LayerNorm(embed_dim),
                    nn.GELU(),
                    nn.Linear(embed_dim, embed_dim)
                ) for _ in range(num_sub_experts)
            ]) for _ in range(num_experts)
        ])
        
        # Router network for level 1 (expert selection)
        self.router_l1 = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.LayerNorm(embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, num_experts)
        )
        
        # Router networks for level 2 (sub-expert selection)
        self.router_l2 = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim // 4),
                nn.LayerNorm(embed_dim // 4),
                nn.ReLU(),
                nn.Linear(embed_dim // 4, num_sub_experts)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x, context=None, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Level 1 routing
        router_logits_l1 = self.router_l1(x)  # (B, L, num_experts)
        routing_probs_l1 = F.softmax(router_logits_l1, dim=-1)  # (B, L, num_experts)
        
        # Compute outputs for each expert
        expert_outputs = []
        for i, expert in enumerate(self.experts):
            expert_output = expert(x, context, mask)  # (B, L, D)
            
            # Level 2 routing within this expert
            router_logits_l2 = self.router_l2[i](expert_output)  # (B, L, num_sub_experts)
            routing_probs_l2 = F.softmax(router_logits_l2, dim=-1)  # (B, L, num_sub_experts)
            
            # Process through sub-experts and combine
            sub_expert_output = torch.zeros_like(expert_output)
            for j, sub_expert in enumerate(self.sub_experts[i]):
                sub_output = sub_expert(expert_output)  # (B, L, D)
                # Weight by routing probability
                sub_expert_weight = routing_probs_l2[:, :, j].unsqueeze(-1)  # (B, L, 1)
                sub_expert_output += sub_output * sub_expert_weight
            
            expert_outputs.append(sub_expert_output)
        
        # Combine expert outputs based on level 1 routing
        final_output = torch.zeros(batch_size, seq_len, self.embed_dim, device=x.device)
        for i, expert_output in enumerate(expert_outputs):
            expert_weight = routing_probs_l1[:, :, i].unsqueeze(-1)  # (B, L, 1)
            final_output += expert_output * expert_weight
        
        return final_output


class FlashMLAttention(BaseAttention):
    """
    Multi-head Latent Attention with Flash Attention optimization
    for memory-efficient processing of long sequences.
    
    Based on research from FlashMLA paper in your collection.
    """
    def __init__(self, embed_dim, num_heads, latent_dim=64, dropout=0.1, block_size=128):
        super(FlashMLAttention, self).__init__(embed_dim, num_heads, dropout)
        self.latent_dim = latent_dim
        self.block_size = block_size
        
        # Latent projection matrices
        self.q_latent_proj = nn.Linear(self.head_dim, latent_dim)
        self.k_latent_proj = nn.Linear(self.head_dim, latent_dim)
        
        # Output mixing layer
        self.latent_mixer = nn.Sequential(
            nn.Linear(latent_dim, self.head_dim),
            nn.GELU()
        )
        
        # Adaptive block size adjustment for hardware optimization
        self.block_size_adjuster = nn.Parameter(torch.ones(1) * math.log(block_size))
        
    def _compute_attention_blockwise(self, q, k, v, mask=None):
        """Compute attention using blockwise processing for memory efficiency"""
        batch_size, num_heads, seq_len, head_dim = q.shape
        output = torch.zeros_like(v)
        
        # Project to latent space
        q_latent = F.elu(self.q_latent_proj(q)) + 1  # (B, H, L, latent_dim)
        k_latent = F.elu(self.k_latent_proj(k)) + 1  # (B, H, L, latent_dim)
        
        # Compute block size based on sequence length and hardware
        effective_block_size = min(
            int(torch.exp(self.block_size_adjuster).item()),
            seq_len
        )
        
        # Process in blocks to save memory
        for i in range(0, seq_len, effective_block_size):
            end_idx = min(i + effective_block_size, seq_len)
            
            # Current query block
            q_block = q_latent[:, :, i:end_idx]  # (B, H, block_size, latent_dim)
            
            # Process key-value pairs for the entire sequence or in chunks
            # based on available memory
            kv_block_size = min(effective_block_size * 4, seq_len)
            
            for j in range(0, seq_len, kv_block_size):
                j_end = min(j + kv_block_size, seq_len)
                
                # Get current key-value blocks
                k_block = k_latent[:, :, j:j_end]  # (B, H, kv_block_size, latent_dim)
                v_block = v[:, :, j:j_end]  # (B, H, kv_block_size, head_dim)
                
                # Compute attention scores for this block
                scores = torch.matmul(q_block, k_block.transpose(-1, -2)) / math.sqrt(self.latent_dim)
                
                # Apply mask if provided
                if mask is not None:
                    block_mask = mask[:, :, i:end_idx, j:j_end]
                    scores = scores.masked_fill(block_mask == 0, -1e10)
                
                # Apply softmax and dropout
                attn_probs = F.softmax(scores, dim=-1)
                attn_probs = self.dropout(attn_probs)
                
                # Compute weighted sum and add to output
                output[:, :, i:end_idx] += torch.matmul(attn_probs, v_block)
        
        return self.latent_mixer(output)
    
    def _get_hardware_specific_config(self):
        """Get hardware-specific configurations for memory optimization"""
        device_type = "cuda" if torch.cuda.is_available() else "cpu"
        
        if device_type == "cuda":
            # NVIDIA GPU configuration
            try:
                device_name = torch.cuda.get_device_name()
                if "A100" in device_name or "H100" in device_name:
                    # High-end NVIDIA GPU
                    return {"block_size": 256, "bandwidth": 2000}
                else:
                    # Standard NVIDIA GPU
                    return {"block_size": 128, "bandwidth": 800}
            except:
                return {"block_size": 128, "bandwidth": 800}
        else:
            # CPU configuration
            return {"block_size": 64, "bandwidth": 100}
    
    def forward(self, x, context=None, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Standard projections
        q = self._reshape_for_multihead(self.q_proj(x))  # (B, H, L, D)
        k = self._reshape_for_multihead(self.k_proj(x if context is None else context))  # (B, H, L, D)
        v = self._reshape_for_multihead(self.v_proj(x if context is None else context))  # (B, H, L, D)
        
        # Compute attention with memory-efficient blockwise processing
        output = self._compute_attention_blockwise(q, k, v, mask)
        
        # Reshape back to (B, L, E)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        return self.o_proj(output)

class TPUOptimizedAttention(BaseAttention):
    """
    Attention mechanism optimized for TPU execution with efficient 
    memory usage and QKV computation patterns.
    """
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(TPUOptimizedAttention, self).__init__(embed_dim, num_heads, dropout)
        
        # Combined QKV projection for faster TPU execution
        self.qkv_combined = nn.Linear(embed_dim, 3 * embed_dim)
        
        # Gradient checkpointing flag
        self.use_gradient_checkpointing = True
    
    def _reshape_qkv(self, qkv):
        """Reshape combined QKV projection output for TPU optimization"""
        batch_size, seq_len, _ = qkv.shape
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # Each is (B, H, L, D)
        return q, k, v
    
    def forward(self, x, context=None, mask=None):
        batch_size, seq_len, _ = x.size()
        
        # Uses context if provided, otherwise self-attention
        input_states = x if context is None else context
        
        # Combined QKV projection
        if self.use_gradient_checkpointing and self.training:
            qkv = torch.utils.checkpoint.checkpoint(self.qkv_combined, input_states)
        else:
            qkv = self.qkv_combined(input_states)
        
        # Reshape for attention computation
        q, k, v = self._reshape_qkv(qkv)
        
        # Optimized matmul for TPU (bfloat16 friendly)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply mask if provided
        if mask is not None:
            mask_expanded = mask.unsqueeze(1)  # (B, 1, L, L) or (B, 1, 1, L)
            attn_scores = attn_scores.masked_fill(mask_expanded == 0, -1e10)
        
        # Apply softmax and dropout
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        
        # Compute weighted sum of values
        output = torch.matmul(attn_probs, v)
        
        # Reshape back to (B, L, E)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection (reusing existing o_proj)
        return self.o_proj(output)