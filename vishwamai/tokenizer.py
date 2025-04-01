import sentencepiece as spm
import numpy as np
from typing import List, Union, Dict, Optional, Tuple
from functools import lru_cache
import os
import json
from huggingface_hub import snapshot_download
from PIL import Image
import base64
import io

class VishwamAITokenizer:
    def __init__(
        self, 
        model_path: str,
        max_length: int = 512,
        pad_token: str = "[PAD]",
        unk_token: str = "[UNK]",
        bos_token: str = "[BOS]",
        eos_token: str = "[EOS]",
        img_token: str = "[IMG]",
        vid_token: str = "[VID]",
        aud_token: str = "[AUD]",
        mask_token: str = "[MASK]",
        sep_token: str = "[SEP]"
    ):
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(model_path)
        self.max_length = max_length
        
        # Special tokens
        self.special_tokens = {
            "pad_token": pad_token,
            "unk_token": unk_token,
            "bos_token": bos_token,
            "eos_token": eos_token,
            "img_token": img_token,
            "vid_token": vid_token,
            "aud_token": aud_token,
            "mask_token": mask_token,
            "sep_token": sep_token
        }
        
        # Initialize multimodal configs
        self.image_size = (224, 224)  # Default image size
        self.max_frames = 32  # Default max video frames
        self.audio_max_length = 30  # Default audio length in seconds
        
        self._init_special_tokens()
        self.token_cache = {}
    
    def _init_special_tokens(self):
        """Initialize special token IDs"""
        for token in self.special_tokens.values():
            if not self.sp.piece_to_id(token):
                print(f"Warning: {token} not found in vocabulary")
    
    @lru_cache(maxsize=1024)
    def encode_image(self, image: Union[str, Image.Image, bytes]) -> Dict[str, Union[List[int], List[str]]]:
        """Encode image input for multimodal processing"""
        if isinstance(image, str):
            # Handle base64 or file path
            if image.startswith('data:image'):
                img_data = base64.b64decode(image.split(',')[1])
                image = Image.open(io.BytesIO(img_data))
            else:
                image = Image.open(image)
        elif isinstance(image, bytes):
            image = Image.open(io.BytesIO(image))
            
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
            
        # Add image token
        tokens = [self.sp.piece_to_id(self.special_tokens["img_token"])]
        
        return {
            "input_ids": tokens,
            "token_type_ids": [1] * len(tokens)  # Use 1 for image tokens
        }

    def encode_multimodal(
        self,
        text: Optional[str] = None,
        image: Optional[Union[str, Image.Image, bytes]] = None,
        audio: Optional[Union[str, bytes]] = None,
        video: Optional[Union[str, bytes]] = None,
        add_special_tokens: bool = True
    ) -> Dict[str, Union[List[int], List[str]]]:
        """Encode multimodal inputs combining text, image, audio and video"""
        input_ids = []
        token_type_ids = []
        
        if text:
            text_tokens = self.sp.encode_as_ids(text)
            input_ids.extend(text_tokens)
            token_type_ids.extend([0] * len(text_tokens))  # 0 for text
            
        if image:
            image_encoding = self.encode_image(image)
            input_ids.extend(image_encoding["input_ids"])
            token_type_ids.extend(image_encoding["token_type_ids"])
            
        if audio:
            # Add audio token
            audio_token = self.sp.piece_to_id(self.special_tokens["aud_token"])
            input_ids.append(audio_token)
            token_type_ids.append(2)  # 2 for audio
            
        if video:
            # Add video token 
            video_token = self.sp.piece_to_id(self.special_tokens["vid_token"])
            input_ids.append(video_token)
            token_type_ids.append(3)  # 3 for video
            
        if add_special_tokens:
            # Add BOS/EOS
            bos_token = self.sp.piece_to_id(self.special_tokens["bos_token"])
            eos_token = self.sp.piece_to_id(self.special_tokens["eos_token"])
            input_ids = [bos_token] + input_ids + [eos_token]
            token_type_ids = [0] + token_type_ids + [0]
            
        # Truncate to max length
        if len(input_ids) > self.max_length:
            input_ids = input_ids[:self.max_length]
            token_type_ids = token_type_ids[:self.max_length]
            
        return {
            "input_ids": input_ids,
            "token_type_ids": token_type_ids
        }

    def decode_multimodal(
        self,
        token_ids: List[int],
        token_type_ids: Optional[List[int]] = None,
        skip_special_tokens: bool = True
    ) -> str:
        """Decode multimodal token ids back to text, preserving special tokens for media"""
        output = []
        for i, token_id in enumerate(token_ids):
            token_type = token_type_ids[i] if token_type_ids else 0
            
            if token_type == 0:  # Text
                piece = self.sp.id_to_piece(token_id)
                if not skip_special_tokens or not piece.startswith('['):
                    output.append(piece)
            elif token_type == 1:  # Image 
                output.append(self.special_tokens["img_token"])
            elif token_type == 2:  # Audio
                output.append(self.special_tokens["aud_token"]) 
            elif token_type == 3:  # Video
                output.append(self.special_tokens["vid_token"])
                
        return "".join(output)
    
    @lru_cache(maxsize=10000)
    def encode(
        self, 
        text: Union[str, List[str]],
        add_special_tokens: bool = True,
        padding: bool = True,
        truncation: bool = True
    ) -> np.ndarray:
        """Encode text with caching and advanced options"""
        if isinstance(text, str):
            text = [text]
            
        encoded = []
        for t in text:
            tokens = self.sp.encode_as_ids(t)
            if add_special_tokens:
                tokens = [self.sp.piece_to_id(self.special_tokens["bos_token"])] + \
                        tokens + \
                        [self.sp.piece_to_id(self.special_tokens["eos_token"])]
            
            if truncation:
                tokens = tokens[:self.max_length]
                
            if padding:
                pad_length = self.max_length - len(tokens)
                if pad_length > 0:
                    tokens.extend([self.sp.piece_to_id(self.special_tokens["pad_token"])] * pad_length)
                    
            encoded.append(tokens)
            
        return np.array(encoded)
    
    def decode(self, ids: Union[List[int], np.ndarray], skip_special_tokens: bool = True) -> str:
        """Decode with special token handling"""
        if isinstance(ids, np.ndarray):
            ids = ids.tolist()
            
        if skip_special_tokens:
            special_ids = [self.sp.piece_to_id(token) for token in self.special_tokens.values()]
            ids = [id for id in ids if id not in special_ids]
            
        return self.sp.decode_ids(ids)
    
    def batch_encode(self, texts: List[str], batch_size: int = 32, **kwargs) -> List[np.ndarray]:
        """Process texts in batches to manage memory"""
        batches = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            encoded_batch = self.encode(batch, **kwargs)
            batches.append(encoded_batch)
        return np.concatenate(batches, axis=0)
    
    @property
    def vocab_size(self) -> int:
        return self.sp.get_piece_size()
    
    def save_pretrained(self, path: str):
        """Save tokenizer configuration"""
        self.sp.save(f"{path}/tokenizer.model")
    
    @classmethod
    def from_pretrained(cls, model_name_or_path: str, **kwargs):
        """Load tokenizer from HuggingFace Hub or local directory"""
        # Download tokenizer if not local
        if not os.path.isdir(model_name_or_path):
            model_path = snapshot_download(
                repo_id=model_name_or_path,
                allow_patterns=["tokenizer.model", "tokenizer.json"],
                local_files_only=False
            )
        else:
            model_path = model_name_or_path
            
        # Find tokenizer model file
        tokenizer_path = os.path.join(model_path, "tokenizer.model")
        if not os.path.exists(tokenizer_path):
            raise ValueError(f"No tokenizer.model found in {model_path}")
            
        return cls(
            model_path=tokenizer_path,
            **kwargs
        )
