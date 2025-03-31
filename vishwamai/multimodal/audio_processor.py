"""Audio processing utilities for multimodal inputs with Whisper integration."""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Optional, Dict, Union, Any
import functools
import whisper
from whisper.audio import log_mel_spectrogram, pad_or_trim, SAMPLE_RATE as WHISPER_SAMPLE_RATE

class AudioProcessor:
    """Processor for preparing audio inputs for the model, with Whisper integration."""

    def __init__(
        self,
        sample_rate: int = WHISPER_SAMPLE_RATE,  # Default to Whisper's 16000 Hz
        n_mels: int = 80,                        # Whisper default
        max_length: Optional[int] = None,        # Max frames (None for Whisper's default ~30s)
        normalize: bool = True,                  # Normalize spectrogram
        pad_to_multiple: Optional[int] = 8,      # Padding for model compatibility
        use_whisper: bool = True                 # Toggle Whisper processing
    ):
        """
        Initialize the AudioProcessor.

        Args:
            sample_rate: Target sampling rate (default matches Whisper).
            n_mels: Number of mel bins (default matches Whisper).
            max_length: Maximum number of frames (None for Whisper's default).
            normalize: Whether to normalize the spectrogram.
            pad_to_multiple: Pad frames to a multiple of this value (e.g., for model alignment).
            use_whisper: Use Whisper's preprocessing if True, else custom mel spectrogram.
        """
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.max_length = max_length
        self.normalize = normalize
        self.pad_to_multiple = pad_to_multiple
        self.use_whisper = use_whisper

        if not use_whisper:
            # Custom mel filterbank parameters (fallback if not using Whisper)
            self.n_fft = 400
            self.hop_length = 160
            self.mel_basis = self._create_mel_filterbank()

    def _create_mel_filterbank(self) -> jnp.ndarray:
        """Create mel filterbank matrix (used only if use_whisper=False)."""
        freq_max = self.sample_rate / 2
        mel_max = 2595.0 * jnp.log10(1.0 + freq_max / 700.0)
        mel_points = jnp.linspace(0, mel_max, self.n_mels + 2)
        hz_points = 700.0 * (10**(mel_points / 2595.0) - 1.0)
        bins = jnp.floor((self.n_fft + 1) * hz_points / self.sample_rate).astype(int)

        fbank = jnp.zeros((self.n_mels, int(self.n_fft / 2 + 1)))
        for i in range(self.n_mels):
            left, center, right = bins[i], bins[i + 1], bins[i + 2]
            for j in range(left, center):
                fbank = fbank.at[i, j].set((j - left) / (center - left))
            for j in range(center, right):
                fbank = fbank.at[i, j].set((right - j) / (right - center))

        return fbank

    @functools.partial(jax.jit, static_argnums=(0,))
    def _stft(self, audio: jnp.ndarray) -> jnp.ndarray:
        """Short-time Fourier transform with optimized memory usage and TPU acceleration."""
        # Use TPU-optimized operations
        window = jnp.hanning(self.n_fft + 1)[:-1]
        pad_width = (self.n_fft - self.hop_length) // 2
        
        # Memory-efficient padding
        audio = jax.lax.dynamic_slice_in_dim(
            jnp.pad(audio, (pad_width, pad_width), mode='reflect'),
            0,
            audio.shape[0] + 2 * pad_width
        )

        # Efficient patch extraction using TPU-optimized convolution
        frames = jax.lax.conv_general_dilated_patches(
            audio[None, None, :],
            filter_shape=(self.n_fft,),
            window_strides=(self.hop_length,),
            padding='VALID',
            dimension_numbers=('NCHW', 'OIHW', 'NCHW')
        )
        
        # Memory-efficient windowing
        frames = frames[0] * window
        
        # Optimized FFT computation
        return jax.vmap(jnp.fft.rfft)(frames)

    @functools.partial(jax.jit, static_argnums=(0,))
    def _compute_mel_spectrogram(
        self,
        audio: jnp.ndarray,
        return_magnitude: bool = True
    ) -> jnp.ndarray:
        """Compute mel spectrogram from audio waveform (used only if use_whisper=False)."""
        stft = self._stft(audio)
        magnitude = jnp.abs(stft)
        power = jnp.square(magnitude)
        mel_spec = jnp.dot(power, self.mel_basis.T)
        if return_magnitude:
            mel_spec = jnp.sqrt(mel_spec)
        if self.normalize:
            mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-5)
        return mel_spec

    def _process_with_whisper(
        self,
        audio: Union[jnp.ndarray, np.ndarray]
    ) -> jnp.ndarray:
        """
        Process audio using Whisper's log mel spectrogram.

        Args:
            audio: Audio waveform [length] or [batch, length].

        Returns:
            Log mel spectrogram as JAX array [batch, frames, n_mels].
        """
        if isinstance(audio, jnp.ndarray):
            audio = np.array(audio)

        if audio.ndim == 1:
            audio = audio[None]

        specs = []
        for waveform in audio:
            # Whisper expects audio at 16000 Hz; resamples if needed internally
            # Pad or trim to Whisper's default length (30s = 480000 samples)
            waveform = pad_or_trim(waveform, length=WHISPER_SAMPLE_RATE * 30)
            # Generate log mel spectrogram (returns [n_mels, frames])
            mel_spec = log_mel_spectrogram(
                waveform,
                n_mels=self.n_mels,
                padding=0  # We'll handle additional padding later
            )
            # Transpose to [frames, n_mels] for consistency
            mel_spec = mel_spec.T
            specs.append(mel_spec)

        return jnp.stack(specs)

    def __call__(
        self,
        audio: Union[jnp.ndarray, np.ndarray],
        sampling_rate: Optional[int] = None,
        return_tensors: bool = True
    ) -> Dict[str, Any]:
        """
        Process audio input.

        Args:
            audio: Audio waveform of shape [length] or [batch, length].
            sampling_rate: Optional sampling rate of input (used only if not using Whisper).
            return_tensors: Whether to return JAX tensors.

        Returns:
            Dictionary containing processed features:
                - input_features: Mel spectrogram [batch, frames, n_mels]
                - sample_rate: Sampling rate used
                - num_frames: Number of frames
                - num_mel_bins: Number of mel bins
        """
        # Convert to JAX array if needed
        if not isinstance(audio, (jnp.ndarray, np.ndarray)):
            audio = np.array(audio)

        if audio.ndim == 1:
            audio = audio[None]

        # Process audio
        if self.use_whisper:
            specs = self._process_with_whisper(audio)
        else:
            if sampling_rate is not None and sampling_rate != self.sample_rate:
                raise NotImplementedError("Resampling not implemented for custom processing.")
            specs = jnp.stack([self._compute_mel_spectrogram(w) for w in audio])

        # Handle maximum length
        if self.max_length is not None:
            if specs.shape[1] > self.max_length:
                specs = specs[:, :self.max_length, :]
            elif specs.shape[1] < self.max_length:
                pad_length = self.max_length - specs.shape[1]
                specs = jnp.pad(specs, ((0, 0), (0, pad_length), (0, 0)), mode='constant')

        # Pad to multiple if specified
        if self.pad_to_multiple is not None:
            target_length = (
                (specs.shape[1] + self.pad_to_multiple - 1) // self.pad_to_multiple * self.pad_to_multiple
            )
            if target_length > specs.shape[1]:
                pad_length = target_length - specs.shape[1]
                specs = jnp.pad(specs, ((0, 0), (0, pad_length), (0, 0)), mode='constant')

        # Convert to NumPy if not returning tensors
        if not return_tensors:
            specs = np.array(specs)

        return {
            'input_features': specs,
            'sample_rate': self.sample_rate,
            'num_frames': specs.shape[1],
            'num_mel_bins': specs.shape[2]
        }