import torch
import torch.nn as nn
from .config import MultimodalConfig
from typing import List, Dict, Optional, Any

class ModalityFusion(nn.Module):
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        
        # Cross-attention layers
        self.fusion_layers = nn.ModuleList([
            CrossModalLayer(config)
            for _ in range(config.fusion_layers)
        ])
        
        # Modality type embeddings
        if config.modality_type_embeddings:
            self.modality_embeddings = nn.Embedding(3, config.fusion_dim)  # text, image, audio
        
        self.norm = nn.LayerNorm(config.fusion_dim)
        
    def forward(
        self,
        text_features: torch.Tensor,
        image_features: Optional[torch.Tensor] = None,
        audio_features: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Prepare features and masks for each modality
        batch_size = text_features.size(0)
        device = text_features.device
        
        # Add modality embeddings if enabled
        if self.config.modality_type_embeddings:
            text_features = text_features + self.modality_embeddings(
                torch.zeros(batch_size, 1, device=device, dtype=torch.long)
            )
            if image_features is not None:
                image_features = image_features + self.modality_embeddings(
                    torch.ones(batch_size, 1, device=device, dtype=torch.long)
                )
            if audio_features is not None:
                audio_features = audio_features + self.modality_embeddings(
                    2 * torch.ones(batch_size, 1, device=device, dtype=torch.long)
                )
        
        # Combine features from available modalities
        features = [text_features]
        if image_features is not None:
            features.append(image_features)
        if audio_features is not None:
            features.append(audio_features)
            
        # Concatenate features
        combined_features = torch.cat(features, dim=1)
        
        # Process through fusion layers
        for layer in self.fusion_layers:
            combined_features = layer(
                combined_features,
                attention_mask=attention_mask
            )
            
        return self.norm(combined_features)

class CrossModalLayer(nn.Module):
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=config.fusion_dim,
            num_heads=config.fusion_heads,
            dropout=0.1
        )
        self.feed_forward = nn.Sequential(
            nn.Linear(config.fusion_dim, config.fusion_dim * 4),
            nn.GELU(),
            nn.Linear(config.fusion_dim * 4, config.fusion_dim)
        )
        self.norm1 = nn.LayerNorm(config.fusion_dim)
        self.norm2 = nn.LayerNorm(config.fusion_dim)
        self.dropout = nn.Dropout(0.1)
        
    def forward(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        # Self-attention
        residual = x
        x = self.norm1(x)
        x, _ = self.attention(x, x, x, attn_mask=attention_mask)
        x = self.dropout(x)
        x = residual + x
        
        # Feed-forward
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = self.dropout(x)
        x = residual + x
        
        return x
