import torch
import torch.nn as nn

class TreeAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super(TreeAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.attention = nn.MultiheadAttention(embed_dim, num_heads)

    def forward(self, query, key, value, key_padding_mask=None, need_weights=True, attn_mask=None):
        return self.attention(query, key, value, key_padding_mask, need_weights, attn_mask)
