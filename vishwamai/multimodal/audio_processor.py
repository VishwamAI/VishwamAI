"""Audio processing utilities for multimodal inputs."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Tuple, Union, Dict, Any
import functools

class AudioProcessor:
    """Processor for preparing audio inputs for the model."""
    
    def __init__(
        self,
        sample_rate: int = 16000,
        n_fft: int = 400,
        hop_length: int = 160,
        n_mels: int = 80,
        max_length: Optional[int] = None,
        normalize: bool = True,
        pad_to_multiple: Optional[int] = 8
    ):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.max_length = max_length
        self.normalize = normalize
        self.pad_to_multiple = pad_to_multiple
        
        # Initialize mel filterbank matrix
        self.mel_basis = self._create_mel_filterbank()
    
    def _create_mel_filterbank(self) -> jnp.ndarray:
        """Create mel filterbank matrix."""
        # Create mel filter matrix
        freq_max = self.sample_rate / 2
        mel_max = 2595.0 * jnp.log10(1.0 + freq_max / 700.0)
        
        # Equally spaced points in mel scale
        mel_points = jnp.linspace(0, mel_max, self.n_mels + 2)
        hz_points = 700.0 * (10**(mel_points / 2595.0) - 1.0)
        
        # Convert Hz points to FFT bin numbers
        bins = jnp.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)
        
        # Create mel filterbank matrix
        fbank = jnp.zeros((self.n_mels, int(self.n_fft / 2 + 1)))
        for i in range(self.n_mels):
            left = bins[i]
            center = bins[i + 1]
            right = bins[i + 2]
            
            # Create triangular filters
            for j in range(left, center):
                fbank = fbank.at[i, j].set(
                    (j - left) / (center - left)
                )
            for j in range(center, right):
                fbank = fbank.at[i, j].set(
                    (right - j) / (right - center)
                )
                
        return fbank
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _stft(self, audio: jnp.ndarray) -> jnp.ndarray:
        """Short-time Fourier transform."""
        # Create Hann window
        window = jnp.hanning(self.n_fft + 1)[:-1]
        
        # Pad signal
        pad_width = (self.n_fft - self.hop_length) // 2
        audio = jnp.pad(audio, (pad_width, pad_width), mode='reflect')
        
        # Extract frames
        frames = jax.lax.conv_general_dilated_patches(
            audio[None, None, :],
            filter_shape=(self.n_fft,),
            window_strides=(self.hop_length,),
            padding='VALID'
        )
        frames = frames[0] * window
        
        # Compute FFT
        stft = jax.vmap(jnp.fft.rfft)(frames)
        return stft
    
    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_mel_spectrogram(
        self,
        audio: jnp.ndarray,
        return_magnitude: bool = True
    ) -> jnp.ndarray:
        """Compute mel spectrogram from audio waveform."""
        # Compute STFT
        stft = self._stft(audio)
        
        # Convert to power spectrogram
        magnitude = jnp.abs(stft)
        power = jnp.square(magnitude)
        
        # Apply mel filterbank
        mel_spec = jnp.dot(power, self.mel_basis.T)
        
        if return_magnitude:
            mel_spec = jnp.sqrt(mel_spec)
            
        if self.normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)
            
        return mel_spec
    
    def __call__(
        self,
        audio: Union[jnp.ndarray, np.ndarray],
        sampling_rate: Optional[int] = None,
        return_tensors: bool = True
    ) -> Dict[str, Any]:
        """Process audio input.
        
        Args:
            audio: Audio waveform of shape [length] or [batch, length]
            sampling_rate: Optional sampling rate of input
            return_tensors: Whether to return jax tensors
            
        Returns:
            Dictionary containing processed features
        """
        # Handle input format
        if not isinstance(audio, jnp.ndarray):
            audio = jnp.array(audio)
            
        if audio.ndim == 1:
            audio = audio[None]
            
        # Resample if needed
        if sampling_rate is not None and sampling_rate != self.sample_rate:
            # Note: Implement resampling if needed
            raise NotImplementedError("Resampling not yet implemented")
            
        # Process each audio in batch
        specs = []
        for waveform in audio:
            # Compute mel spectrogram
            mel_spec = self._compute_mel_spectrogram(waveform)
            
            # Handle maximum length
            if self.max_length is not None:
                if mel_spec.shape[0] > self.max_length:
                    mel_spec = mel_spec[:self.max_length]
                elif mel_spec.shape[0] < self.max_length:
                    pad_length = self.max_length - mel_spec.shape[0]
                    mel_spec = jnp.pad(
                        mel_spec,
                        ((0, pad_length), (0, 0)),
                        mode='constant'
                    )
            
            # Pad to multiple if needed
            if self.pad_to_multiple is not None:
                target_length = (
                    (mel_spec.shape[0] + self.pad_to_multiple - 1)
                    // self.pad_to_multiple
                    * self.pad_to_multiple
                )
                if target_length > mel_spec.shape[0]:
                    pad_length = target_length - mel_spec.shape[0]
                    mel_spec = jnp.pad(
                        mel_spec,
                        ((0, pad_length), (0, 0)),
                        mode='constant'
                    )
                    
            specs.append(mel_spec)
            
        # Stack batch
        specs = jnp.stack(specs)
        
        if not return_tensors:
            specs = np.array(specs)
            
        return {
            'input_features': specs,
            'sample_rate': self.sample_rate,
            'num_frames': specs.shape[1],
            'num_mel_bins': specs.shape[2]
        }