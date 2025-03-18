"""Configuration and initialization utilities for multimodal VishwamAI."""

from dataclasses import dataclass
from typing import Optional, Dict, Any, List
import json

@dataclass
class VisionConfig:
    """Vision model configuration."""
    num_layers: int = 12
    image_size: int = 896
    patch_size: int = 14
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

@dataclass
class AudioConfig:
    """Audio model configuration."""
    num_layers: int = 12
    hidden_size: int = 1024
    num_attention_heads: int = 16
    intermediate_size: int = 4096
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 80
    max_length: Optional[int] = None
    normalize: bool = True
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1

@dataclass
class SonarConfig:
    """SONAR model configuration."""
    num_languages: int = 200
    embedding_dim: int = 1024
    use_speech: bool = True
    use_text_decoder: bool = True
    fairseq2_model_path: Optional[str] = None
    language_codes: Optional[List[str]] = None

@dataclass
class MultimodalConfig:
    """Configuration for multimodal model."""
    vision_config: Optional[VisionConfig] = None
    audio_config: Optional[AudioConfig] = None
    sonar_config: Optional[SonarConfig] = None
    hidden_size: int = 4096
    fusion_layers: int = 2
    num_attention_heads: int = 32
    intermediate_size: int = 16384
    max_position_embeddings: int = 4096
    fusion_dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    layer_norm_epsilon: float = 1e-5
    initializer_range: float = 0.02

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'MultimodalConfig':
        """Create config from dictionary."""
        # Handle nested configs
        if 'vision_config' in config_dict:
            config_dict['vision_config'] = VisionConfig(
                **config_dict['vision_config']
            )
        if 'audio_config' in config_dict:
            config_dict['audio_config'] = AudioConfig(
                **config_dict['audio_config']
            )
        if 'sonar_config' in config_dict:
            config_dict['sonar_config'] = SonarConfig(
                **config_dict['sonar_config']
            )
        return cls(**config_dict)

    @classmethod
    def from_json_file(cls, json_file: str) -> 'MultimodalConfig':
        """Load config from JSON file."""
        with open(json_file, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        config_dict = {
            'hidden_size': self.hidden_size,
            'fusion_layers': self.fusion_layers,
            'num_attention_heads': self.num_attention_heads,
            'intermediate_size': self.intermediate_size,
            'max_position_embeddings': self.max_position_embeddings,
            'fusion_dropout_rate': self.fusion_dropout_rate,
            'attention_dropout_rate': self.attention_dropout_rate,
            'layer_norm_epsilon': self.layer_norm_epsilon,
            'initializer_range': self.initializer_range
        }
        
        if self.vision_config:
            config_dict['vision_config'] = {
                'num_layers': self.vision_config.num_layers,
                'image_size': self.vision_config.image_size,
                'patch_size': self.vision_config.patch_size,
                'hidden_size': self.vision_config.hidden_size,
                'num_attention_heads': self.vision_config.num_attention_heads,
                'intermediate_size': self.vision_config.intermediate_size,
                'dropout_rate': self.vision_config.dropout_rate,
                'attention_dropout_rate': self.vision_config.attention_dropout_rate
            }
            
        if self.audio_config:
            config_dict['audio_config'] = {
                'num_layers': self.audio_config.num_layers,
                'hidden_size': self.audio_config.hidden_size,
                'num_attention_heads': self.audio_config.num_attention_heads,
                'intermediate_size': self.audio_config.intermediate_size,
                'sample_rate': self.audio_config.sample_rate,
                'n_fft': self.audio_config.n_fft,
                'hop_length': self.audio_config.hop_length,
                'n_mels': self.audio_config.n_mels,
                'max_length': self.audio_config.max_length,
                'normalize': self.audio_config.normalize,
                'dropout_rate': self.audio_config.dropout_rate,
                'attention_dropout_rate': self.audio_config.attention_dropout_rate
            }
        
        if self.sonar_config:
            config_dict['sonar_config'] = {
                'num_languages': self.sonar_config.num_languages,
                'embedding_dim': self.sonar_config.embedding_dim,
                'use_speech': self.sonar_config.use_speech,
                'use_text_decoder': self.sonar_config.use_text_decoder,
                'fairseq2_model_path': self.sonar_config.fairseq2_model_path,
                'language_codes': self.sonar_config.language_codes
            }
            
        return config_dict

    def save_pretrained(self, save_directory: str):
        """Save config to JSON file."""
        config_dict = self.to_dict()
        output_file = f"{save_directory}/multimodal_config.json"
        
        with open(output_file, 'w') as f:
            json.dump(config_dict, f, indent=2)

def create_default_multimodal_config(
    include_vision: bool = True,
    include_audio: bool = True,
    include_sonar: bool = False,
    **kwargs
) -> MultimodalConfig:
    """Create default multimodal configuration.
    
    Args:
        include_vision: Whether to include vision config
        include_audio: Whether to include audio config
        include_sonar: Whether to include sonar config
        **kwargs: Override default config values
        
    Returns:
        MultimodalConfig instance
    """
    config_dict = {
        'hidden_size': 4096,
        'fusion_layers': 2,
        'num_attention_heads': 32,
        'intermediate_size': 16384,
        'max_position_embeddings': 4096,
        'fusion_dropout_rate': 0.1,
        'attention_dropout_rate': 0.1,
        'layer_norm_epsilon': 1e-5,
        'initializer_range': 0.02
    }
    
    if include_vision:
        config_dict['vision_config'] = {
            'num_layers': 12,
            'image_size': 896,
            'patch_size': 14,
            'hidden_size': 1024,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'dropout_rate': 0.1,
            'attention_dropout_rate': 0.1
        }
        
    if include_audio:
        config_dict['audio_config'] = {
            'num_layers': 12,
            'hidden_size': 1024,
            'num_attention_heads': 16,
            'intermediate_size': 4096,
            'sample_rate': 16000,
            'n_fft': 400,
            'hop_length': 160,
            'n_mels': 80,
            'normalize': True,
            'dropout_rate': 0.1,
            'attention_dropout_rate': 0.1
        }
    
    if include_sonar:
        config_dict['sonar_config'] = {
            'num_languages': 200,
            'embedding_dim': 1024,
            'use_speech': True,
            'use_text_decoder': True,
            'fairseq2_model_path': None,
            'language_codes': None
        }
    
    # Override defaults with any provided values
    config_dict.update(kwargs)
    
    return MultimodalConfig.from_dict(config_dict)