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
class TPUConfig:
    """TPU-specific configuration."""
    num_devices: int = 8  # Number of TPU cores
    memory_limit: int = 16 * 1024**3  # 16GB per TPU core
    preferred_dtype: str = 'bfloat16'  # Best precision for TPU matrix ops
    optimal_batch_size: int = 1024  # For good TPU utilization
    optimal_sequence_length: int = 2048  # Optimal sequence length
    use_kernel_fusion: bool = True  # Enable kernel fusion optimizations
    use_gradient_checkpointing: bool = True  # Enable gradient checkpointing
    optimal_shard_size: int = 128  # Size for optimal TPU sharding

@dataclass
class MultimodalConfig:
    """Configuration for multimodal model."""
    vision: VisionConfig = VisionConfig()
    audio: AudioConfig = AudioConfig()
    fusion: FusionConfig = FusionConfig()
    tpu: TPUConfig = TPUConfig()
    
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
        self.block_size = min(self.block_size, self.tpu.optimal_shard_size)
        self.vision.block_size = self.block_size
        self.audio.block_size = self.block_size
        self.fusion.block_size = self.block_size
        
        # Set TPU-optimized dtype
        if self.use_bfloat16:
            self.vision.dtype = 'bfloat16'
            self.audio.dtype = 'bfloat16'
            self.fusion.dtype = 'bfloat16'
            
        # Enable kernel fusion if requested
        if self.use_kernel_fusion:
            self.vision.use_kernel_fusion = True
            self.audio.use_kernel_fusion = True
            self.fusion.use_kernel_fusion = True
            
        # Enable gradient checkpointing if requested
        if self.gradient_checkpointing:
            self.vision.gradient_checkpointing = True
            self.fusion.gradient_checkpointing = True

def create_default_multimodal_config(
    include_vision: bool = True,
    include_audio: bool = True,
    include_sonar: bool = False,
    use_tpu: bool = True
) -> MultimodalConfig:
    """Create default multimodal configuration.
    
    Args:
        include_vision: Whether to include vision components
        include_audio: Whether to include audio components
        include_sonar: Whether to include SONAR components
        use_tpu: Whether to enable TPU optimizations
        
    Returns:
        MultimodalConfig instance
    """
    config = MultimodalConfig()
    
    if not include_vision:
        config.vision = None
        
    if not include_audio:
        config.audio = None
        
    if include_sonar:
        config.sonar = {
            "hidden_dim": 1024,
            "num_layers": 12,
            "num_heads": 16,
            "max_position_embeddings": 2048
        }
        
    if not use_tpu:
        config.use_bfloat16 = False
        config.use_kernel_fusion = False
        config.tpu = None
        
    return config