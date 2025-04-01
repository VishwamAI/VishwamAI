"""SONAR integration for multilingual and multimodal embeddings."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from fairseq2.data.text import (
    StrSplitter,
    StrToIntConverter, 
    StrToTensorConverter,
    LineEnding,
    read_text
)
from vishwamai.layers.layers import TPUGEMMLinear, TPULayerNorm
from vishwamai.layers.attention import FlashAttention

class SonarEncoder(nn.Module):
    """SONAR encoder for multilingual text and speech embeddings."""
    
    hidden_dim: int = 1024
    num_layers: int = 12
    num_heads: int = 16
    mlp_dim: Optional[int] = None
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    use_flash_attention: bool = True
    block_size: int = 64
    dtype: Any = jnp.float32
    max_position_embeddings: int = 2048
    
    def setup(self):
        """Initialize SONAR components."""
        # Text processing components
        self.text_converter = StrToTensorConverter()
        self.str_splitter = StrSplitter()
        self.int_converter = StrToIntConverter()
        
        # Embeddings and position encodings
        self.token_embedding = TPUGEMMLinear(
            features=self.hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(0.02)
        )
        
        # Language embeddings
        self.language_embedding = self.param(
            'language_embedding',
            nn.initializers.normal(0.02),
            (100, self.hidden_dim),  # Support up to 100 languages
            self.dtype
        )
        
        # Transformer layers with memory optimization
        self.encoder_layers = [
            TransformerLayer(
                hidden_dim=self.hidden_dim,
                num_heads=self.num_heads,
                mlp_dim=self.mlp_dim or self.hidden_dim * 4,
                dropout_rate=self.dropout_rate,
                attention_dropout_rate=self.attention_dropout_rate,
                use_flash_attention=self.use_flash_attention,
                block_size=self.block_size,
                dtype=self.dtype
            ) for _ in range(self.num_layers)
        ]
        
        # Output projection
        self.output_norm = TPULayerNorm(dtype=self.dtype)
        self.output_projection = TPUGEMMLinear(
            features=self.hidden_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.normal(0.02)
        )

    def _process_sequence(
        self,
        inputs: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        training: bool = False
    ) -> jnp.ndarray:
        """Process sequence through transformer layers with memory optimizations."""
        x = inputs
        
        for layer in self.encoder_layers:
            x = layer(
                x,
                attention_mask=attention_mask,
                deterministic=not training
            )
            
        x = self.output_norm(x)
        return self.output_projection(x)

    def embed_text(
        self,
        text: List[str],
        src_lang: str,
        return_attn_mask: bool = False,
        training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Embed text sequences with TPU optimizations.
        
        Args:
            text: List of text inputs
            src_lang: Source language code
            return_attn_mask: Whether to return attention mask
            training: Whether in training mode
            
        Returns:
            Dict with text embeddings and optional mask
        """
        # Split and convert text efficiently
        tokens = [self.str_splitter(t) for t in text]
        token_ids = [self.int_converter(t) for t in tokens]
        
        # Convert to tensor embeddings
        embeddings = self.text_converter(token_ids)
        embeddings = jnp.array(embeddings)
        
        # Add language embedding
        lang_idx = hash(src_lang) % 100  # Simple language ID hashing
        lang_embedding = self.language_embedding[lang_idx]
        embeddings = embeddings + lang_embedding
        
        # Create attention mask if needed
        attention_mask = None
        if return_attn_mask:
            max_len = embeddings.shape[1]
            attention_mask = jnp.zeros((len(text), max_len))
            for i, t in enumerate(token_ids):
                attention_mask = attention_mask.at[i, :len(t)].set(1)
        
        # Process through transformer
        outputs = self._process_sequence(
            embeddings,
            attention_mask=attention_mask,
            training=training
        )
        
        return_dict = {
            "text_embeddings": outputs
        }
        if return_attn_mask:
            return_dict["attention_mask"] = attention_mask
            
        return return_dict

    def embed_speech(
        self,
        speech: jnp.ndarray,
        src_lang: str,
        sampling_rate: int = 16000,
        return_attn_mask: bool = False,
        training: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Embed speech inputs with TPU optimizations.
        
        Args:
            speech: Speech waveforms [batch, samples] or [samples]
            src_lang: Source language code
            sampling_rate: Audio sampling rate
            return_attn_mask: Whether to return attention mask
            training: Whether in training mode
            
        Returns:
            Dict with speech embeddings and optional mask
        """
        if speech.ndim == 1:
            speech = speech[None, :]
            
        # Extract features efficiently using TPU
        frame_length = int(sampling_rate * 0.025)  # 25ms frames
        features = jax.vmap(lambda x: jnp.abs(jnp.fft.rfft(x[:frame_length])))(
            jnp.array_split(speech, speech.shape[1] // frame_length, axis=1)
        )
        
        # Add language embedding
        lang_idx = hash(src_lang) % 100
        lang_embedding = self.language_embedding[lang_idx]
        features = features + lang_embedding
        
        # Create attention mask if needed
        attention_mask = None
        if return_attn_mask:
            attention_mask = jnp.ones(features.shape[:2])
            
        # Process through transformer
        outputs = self._process_sequence(
            features,
            attention_mask=attention_mask,
            training=training
        )
        
        return_dict = {
            "speech_embeddings": outputs
        }
        if return_attn_mask:
            return_dict["attention_mask"] = attention_mask
            
        return return_dict

    def decode_text(
        self,
        embeddings: jnp.ndarray,
        tgt_lang: str,
        max_length: int = 128
    ) -> List[str]:
        """Decode embeddings to target language text.
        
        Args:
            embeddings: Input embeddings [batch, seq_len, dim]
            tgt_lang: Target language code 
            max_length: Maximum output sequence length
            
        Returns:
            List of decoded text sequences
        """
        # Project to vocabulary efficiently
        logits = self.output_projection(embeddings)
        
        # Convert to token indices
        token_ids = jnp.argmax(logits, axis=-1)
        
        # Decode sequences
        decoded = []
        for seq in token_ids:
            tokens = [str(t) for t in seq[:max_length]]
            decoded.append(" ".join(tokens))
            
        return decoded

    def get_similarity_score(
        self,
        emb1: jnp.ndarray,
        emb2: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute cosine similarity between embeddings efficiently.
        
        Args:
            emb1: First embeddings [batch, dim]
            emb2: Second embeddings [batch, dim]
            
        Returns:
            Similarity scores [batch]
        """
        # Normalize embeddings
        emb1_norm = emb1 / jnp.linalg.norm(emb1, axis=-1, keepdims=True)
        emb2_norm = emb2 / jnp.linalg.norm(emb2, axis=-1, keepdims=True)
        
        # Compute similarity
        return jnp.sum(emb1_norm * emb2_norm, axis=-1)

    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        speech_input: Optional[jnp.ndarray] = None,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        training: bool = False,
        return_dict: bool = True
    ) -> Dict[str, jnp.ndarray]:
        """Process inputs through text/speech encoder.
        
        Args:
            input_ids: Optional text input ids
            attention_mask: Optional attention mask for text
            speech_input: Optional speech input
            src_lang: Source language code
            tgt_lang: Optional target language for decoding
            training: Whether in training mode
            return_dict: Whether to return dictionary
            
        Returns:
            Dict with processed outputs
        """
        outputs = {}
        
        if input_ids is not None:
            text_outputs = self.embed_text(
                [str(ids) for ids in input_ids],
                src_lang=src_lang,
                return_attn_mask=True,
                training=training
            )
            outputs.update(text_outputs)
            
        if speech_input is not None:
            speech_outputs = self.embed_speech(
                speech_input,
                src_lang=src_lang,
                return_attn_mask=True,
                training=training
            )
            outputs.update(speech_outputs)
            
        if tgt_lang is not None and "text_embeddings" in outputs:
            decoded_text = self.decode_text(
                outputs["text_embeddings"],
                tgt_lang=tgt_lang
            )
            outputs["decoded_text"] = decoded_text
            
        return outputs if return_dict else list(outputs.values())

class TransformerLayer(nn.Module):
    """Memory-optimized transformer layer for SONAR."""
    
    hidden_dim: int
    num_heads: int
    mlp_dim: int
    dropout_rate: float = 0.1
    attention_dropout_rate: float = 0.1
    use_flash_attention: bool = True
    block_size: int = 64
    dtype: Any = jnp.float32
    
    @nn.compact
    def __call__(
        self,
        x: jnp.ndarray,
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> jnp.ndarray:
        """Process inputs through transformer layer."""
        # Pre-norm
        y = TPULayerNorm(dtype=self.dtype)(x)
        
        # Multi-head attention with memory optimizations
        attention = FlashAttention(
            num_heads=self.num_heads,
            head_dim=self.hidden_dim // self.num_heads,
            dropout_rate=self.attention_dropout_rate,
            dtype=self.dtype,
            use_causal_mask=False,
            block_size=self.block_size
        )(y, mask=attention_mask, deterministic=deterministic)
        
        # Residual connection with gradient checkpointing
        if not deterministic:
            x = nn.remat(lambda x, y: x + y)(x, attention)
        else:
            x = x + attention
            
        # Feed-forward network
        y = TPULayerNorm(dtype=self.dtype)(x)
        y = nn.Sequential([
            TPUGEMMLinear(features=self.mlp_dim, dtype=self.dtype),
            nn.gelu,
            nn.Dropout(rate=self.dropout_rate),
            TPUGEMMLinear(features=self.hidden_dim, dtype=self.dtype),
            nn.Dropout(rate=self.dropout_rate)
        ])(y, deterministic=deterministic)
        
        # Final residual
        x = x + y
        
        return x