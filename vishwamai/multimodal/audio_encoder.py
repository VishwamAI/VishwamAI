import torch
import torch.nn as nn
import torchaudio
from .config import AudioConfig

class AudioPreprocessor(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        
        self.mel_spec = torchaudio.transforms.MelSpectrogram(
            sample_rate=config.sample_rate,
            n_fft=config.n_fft,
            hop_length=config.hop_length,
            n_mels=config.mel_bins
        )
        
        self.spec_augment = torchaudio.transforms.FrequencyMasking(27)
        self.to_db = torchaudio.transforms.AmplitudeToDB()
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        mel = self.mel_spec(audio)
        mel = self.to_db(mel)
        return mel

class AudioEncoder(nn.Module):
    def __init__(self, config: AudioConfig):
        super().__init__()
        self.config = config
        
        self.preprocessor = AudioPreprocessor(config)
        
        # Convolutional layers for processing mel spectrograms
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, config.hidden_size, 3, padding=1)
        )
        
        # Position embeddings
        self.position_embeddings = nn.Parameter(
            torch.zeros(1, config.max_audio_length, config.hidden_size)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=config.hidden_size,
                nhead=config.num_heads,
                dim_feedforward=config.hidden_size * 4,
                dropout=0.1
            )
            for _ in range(config.num_layers)
        ])
        
        self.norm = nn.LayerNorm(config.hidden_size)
        
    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        # Process audio to mel spectrograms
        mel = self.preprocessor(audio)
        
        # Process through conv layers
        features = self.conv(mel.unsqueeze(1))
        features = features.flatten(2).transpose(1, 2)
        
        # Add position embeddings
        features = features + self.position_embeddings[:, :features.size(1)]
        
        # Process through transformer layers
        for layer in self.layers:
            features = layer(features)
            
        return self.norm(features)
