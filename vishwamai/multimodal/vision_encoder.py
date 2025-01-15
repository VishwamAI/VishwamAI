import torch
import torch.nn as nn
from .config import VisionConfig

class PatchEmbedding(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        patch_size = config.patch_size
        
        self.proj = nn.Conv2d(
            config.num_channels,
            config.hidden_size,
            kernel_size=patch_size,
            stride=patch_size
        )
        self.norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, height, width = pixel_values.shape
        embeddings = self.proj(pixel_values)
        embeddings = embeddings.flatten(2).transpose(1, 2)
        embeddings = self.norm(embeddings)
        return embeddings

class VisionEncoder(nn.Module):
    def __init__(self, config: VisionConfig):
        super().__init__()
        self.config = config
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(config)
        
        # Position embeddings
        num_patches = (config.image_size // config.patch_size) ** 2
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, num_patches, config.hidden_size)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.intermediate_size,
                dropout=config.dropout,
                activation='gelu'
            )
            for _ in range(config.num_layers)
        ])
        
        self.dropout = nn.Dropout(config.dropout)
        self.norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        embeddings = self.patch_embed(pixel_values)
        embeddings = embeddings + self.position_embeddings
        embeddings = self.dropout(embeddings)
        
        for layer in self.layers:
            embeddings = layer(embeddings)
            
        return self.norm(embeddings)
