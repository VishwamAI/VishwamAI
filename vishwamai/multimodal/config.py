from dataclasses import dataclass
from typing import Optional, List, Tuple
from ..config import VishwamaiConfig

@dataclass
class VisionConfig:
    image_size: int = 224
    patch_size: int = 16
    num_channels: int = 3
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12
    intermediate_size: int = 3072
    dropout: float = 0.1
    attention_dropout: float = 0.1
    max_image_length: int = (224 // 16) ** 2  # 14x14 patches for 224 image

@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    win_length: int = 400
    mel_bins: int = 80
    hidden_size: int = 768
    num_layers: int = 12
    num_heads: int = 12

@dataclass
class MultimodalConfig(VishwamaiConfig):
    vision_config: VisionConfig = VisionConfig()
    audio_config: AudioConfig = AudioConfig()
    fusion_layers: int = 4
    fusion_heads: int = 8
    fusion_dim: int = 1024
    max_image_length: int = 196  # 14x14 patches for 224 image
    max_audio_length: int = 1024
    modality_type_embeddings: bool = True
