# VishwamAI/development/feature_experiments/new_attention.py
import torch
from torch import nn

class PathAwareAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, num_paths=5):
        super(PathAwareAttention, self).__init__()
        self.num_paths = num_paths
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)
        self.gate = nn.Sequential(
            nn.Linear(embed_dim * num_paths, num_paths),
            nn.Softmax(dim=-1)
        )

    def forward(self, query, key, value, paths):
        # paths: (num_paths, seq_len, embed_dim)
        attentions = []
        for path in paths:
            attn_output, _ = self.attention(query, path, path)
            attentions.append(attn_output)

        # Stack: (seq_len, batch_size, embed_dim, num_paths)
        attentions = torch.stack(attentions, dim=-1)

        # Compute gating weights
        gate_input = attentions.view(attentions.size(0), attentions.size(1), -1)
        gate_weights = self.gate(gate_input).unsqueeze(2)  # (seq_len, batch_size, 1, num_paths)

        # Combine attentions
        weighted_attentions = (attentions * gate_weights).sum(dim=-1)

        return weighted_attentions

# Usage: path_aware_attention = PathAwareAttention(embed_dim=768, num_heads=8)