"""SONAR integration for multilingual and multimodal embeddings."""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass

class SonarEncoder(nn.Module):
    """SONAR encoder for multilingual text and speech embeddings."""
    
    config: Dict[str, Any]
    
    def setup(self):
        # Lazy import fairseq2 to avoid startup dependency
        try:
            import fairseq2.models.sonar
            from fairseq2.models.sonar import SonarModel
            from fairseq2.data.text import TextTokenizer
            
            self.sonar = SonarModel(self.config["fairseq2_model_path"])
            self.tokenizer = TextTokenizer()
            self._has_sonar = True
        except ImportError:
            print("Warning: fairseq2 not found, some features will be disabled")
            self._has_sonar = False
    
    def embed_text(
        self,
        text: List[str],
        src_lang: str,
        return_attn_mask: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Embed text sequences into fixed-size vectors.
        
        Args:
            text: List of text sequences
            src_lang: Source language code
            return_attn_mask: Whether to return attention mask
            
        Returns:
            Dictionary with embeddings and optional attention mask
        """
        if not self._has_sonar:
            raise ImportError("fairseq2 is required for text embedding")
            
        # Tokenize text
        tokens = self.tokenizer.encode(text, language=src_lang)
        
        # Get embeddings from SONAR
        embeddings = self.sonar.encode_text(tokens)
        
        outputs = {"embeddings": embeddings}
        
        if return_attn_mask:
            # Create attention mask
            attention_mask = jnp.ones(embeddings.shape[:2])
            outputs["attention_mask"] = attention_mask
            
        return outputs
    
    def embed_speech(
        self,
        speech: jnp.ndarray,
        src_lang: str,
        sampling_rate: int = 16000,
        return_attn_mask: bool = False
    ) -> Dict[str, jnp.ndarray]:
        """Embed speech segments into fixed-size vectors.
        
        Args:
            speech: Speech waveform array
            src_lang: Source language code
            sampling_rate: Audio sampling rate
            return_attn_mask: Whether to return attention mask
            
        Returns:
            Dictionary with embeddings and optional attention mask
        """
        if not self._has_sonar:
            raise ImportError("fairseq2 is required for speech embedding")
            
        # Get speech embeddings from SONAR
        embeddings = self.sonar.encode_speech(
            speech,
            sample_rate=sampling_rate,
            language=src_lang
        )
        
        outputs = {"embeddings": embeddings}
        
        if return_attn_mask:
            # Create attention mask
            attention_mask = jnp.ones(embeddings.shape[:2])
            outputs["attention_mask"] = attention_mask
            
        return outputs
    
    def decode_text(
        self,
        embeddings: jnp.ndarray,
        tgt_lang: str,
        max_length: int = 128
    ) -> List[str]:
        """Decode embeddings back to text.
        
        Args:
            embeddings: Input embeddings
            tgt_lang: Target language code
            max_length: Maximum sequence length
            
        Returns:
            List of decoded text sequences
        """
        if not self._has_sonar:
            raise ImportError("fairseq2 is required for text decoding")
            
        # Decode embeddings using SONAR
        decoded = self.sonar.decode_text(
            embeddings,
            language=tgt_lang,
            max_len=max_length
        )
        
        return decoded
    
    def get_similarity_score(
        self,
        emb1: jnp.ndarray,
        emb2: jnp.ndarray
    ) -> jnp.ndarray:
        """Compute similarity score between embeddings.
        
        Args:
            emb1: First embedding
            emb2: Second embedding
            
        Returns:
            Similarity score
        """
        # Normalize embeddings
        emb1 = emb1 / jnp.linalg.norm(emb1, axis=-1, keepdims=True)
        emb2 = emb2 / jnp.linalg.norm(emb2, axis=-1, keepdims=True)
        
        # Compute cosine similarity
        return jnp.sum(emb1 * emb2, axis=-1)
    
    def __call__(
        self,
        input_ids: Optional[jnp.ndarray] = None,
        attention_mask: Optional[jnp.ndarray] = None,
        speech_input: Optional[jnp.ndarray] = None,
        src_lang: Optional[str] = None,
        tgt_lang: Optional[str] = None,
        return_dict: bool = True
    ) -> Dict[str, jnp.ndarray]:
        """Forward pass for either text or speech input.
        
        Args:
            input_ids: Text input token ids
            attention_mask: Text attention mask
            speech_input: Speech waveform input
            src_lang: Source language code
            tgt_lang: Target language code (for text decoding)
            return_dict: Whether to return dictionary
            
        Returns:
            Dictionary with embeddings and model outputs
        """
        outputs = {}
        
        if input_ids is not None:
            # Process text input
            text_outputs = self.embed_text(
                input_ids,
                src_lang=src_lang,
                return_attn_mask=True
            )
            outputs.update(text_outputs)
            
        if speech_input is not None:
            # Process speech input
            speech_outputs = self.embed_speech(
                speech_input,
                src_lang=src_lang,
                return_attn_mask=True
            )
            outputs.update(speech_outputs)
            
        if tgt_lang is not None and "embeddings" in outputs:
            # Decode to target language if specified
            decoded_text = self.decode_text(
                outputs["embeddings"],
                tgt_lang=tgt_lang
            )
            outputs["decoded_text"] = decoded_text
            
        return outputs