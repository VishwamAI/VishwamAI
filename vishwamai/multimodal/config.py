"""Configuration and initialization utilities for multimodal VishwamAI."""
from dataclasses import dataclass
from typing import Optional

@dataclass 
class VisionConfig:
    """Configuration for vision encoder."""
    hidden_dim: int = 1024
    num_layers: int = 24
    num_heads: int = 16
    mlp_dim: int = 4096
    patch_size: int = 16
    image_size: int = 224
    use_flash_attention: bool = True
    block_size: int = 64
    dtype: str = 'float32'
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    use_kernel_fusion: bool = True
    gradient_checkpointing: bool = True

@dataclass
class AudioConfig:
    """Configuration for audio processing."""
    sample_rate: int = 16000
    n_mels: int = 128
    n_fft: int = 400
    hop_length: int = 160
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 12
    use_flash_attention: bool = True
    block_size: int = 64
    dtype: str = 'float32'
    dropout_rate: float = 0.1
    use_kernel_fusion: bool = True

@dataclass
class FusionConfig:
    """Configuration for multimodal fusion."""
    hidden_dim: int = 1024
    num_heads: int = 16
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    use_flash_attention: bool = True
    block_size: int = 64
    use_kernel_fusion: bool = True
    gradient_checkpointing: bool = True
    dtype: str = 'float32'

@dataclass
class MultimodalConfig:
    """Configuration for multimodal model."""
    vision: VisionConfig = VisionConfig()
    audio: AudioConfig = AudioConfig()
    fusion: FusionConfig = FusionConfig()
    
    output_dim: int = 512
    use_bfloat16: bool = True
    block_size: int = 64
    use_flash_attention: bool = True
    use_kernel_fusion: bool = True
    gradient_checkpointing: bool = True
    
    def __post_init__(self):
        """Validate and adjust configurations."""
        # Ensure compatible hidden dimensions
        assert self.vision.hidden_dim == self.fusion.hidden_dim, \
            "Vision and fusion hidden dims must match"
        assert self.audio.hidden_dim == self.fusion.hidden_dim, \
            "Audio and fusion hidden dims must match"
            
        # Adjust block sizes for hardware efficiency
        self.block_size = min(self.block_size, 128)  # Max efficient block size
        self.vision.block_size = self.block_size
        self.audio.block_size = self.block_size
        self.fusion.block_size = self.block_size