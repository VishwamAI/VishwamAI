import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union
import os
import gc
import json
import logging
import random

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from functools import partial
from einops import rearrange, repeat
from huggingface_hub import snapshot_download
import safetensors.flax as stf

from google.cloud import storage

# Define ModelArgs before using it
@dataclass
class ModelArgs:
    dim: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    vocab_size: int
    multiple_of: int
    norm_eps: float
    max_batch_size: int
    max_seq_len: int
    n_experts: int
    expert_dim: int
    expert_capacity_factor: float
    window_size: int
    global_tokens: int
    attention_dropout: float
    dropout_rate: float
    expert_dropout: float
    param_dtype: jnp.dtype
    use_rope: bool
    use_flash_attention: bool
    use_alibi: bool
    use_dualpipe: bool
    use_eplb: bool
    use_deepgemm: bool

# Import or define TransformerBlock
class TransformerBlock(nn.Module):
    """TPU-optimized Transformer block."""
    config: ModelArgs

    def setup(self):
        self.layer_norm1 = LayerNorm(epsilon=self.config.norm_eps, dtype=self.config.param_dtype)
        self.self_attention = nn.SelfAttention(
            num_heads=self.config.n_heads,
            dtype=self.config.param_dtype,
            dropout_rate=self.config.attention_dropout,
            deterministic=True
        )
        self.layer_norm2 = LayerNorm(epsilon=self.config.norm_eps, dtype=self.config.param_dtype)
        self.feed_forward = nn.Dense(
            features=self.config.ffn_dim_multiplier * self.config.dim if self.config.ffn_dim_multiplier else self.config.dim,
            dtype=self.config.param_dtype
        )

    def __call__(self, x, attention_mask, deterministic):
        y = self.layer_norm1(x)
        y = self.self_attention(y, attention_mask, deterministic=deterministic)
        y = x + y
        z = self.layer_norm2(y)
        z = self.feed_forward(z)
        return y + z

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# TPU-optimized optimizer creation
@partial(jax.jit, static_argnums=(1, 2, 3, 4, 5))
def create_learning_rate_schedule(
    step: int,
    learning_rate: float = 1e-4,
    warmup_steps: int = 2000,
    num_train_steps: int = 100000,
    weight_decay: float = 0.01,
    num_training_devices: int = 1
) -> float:
    """Create learning rate schedule optimized for TPU."""
    warmup_fn = lambda step: step / max(1, warmup_steps)
    decay_fn = lambda step: 1.0 - (step - warmup_steps) / (num_train_steps - warmup_steps)
    lr = jnp.where(
        step < warmup_steps,
        learning_rate * warmup_fn(step),
        learning_rate * decay_fn(step)
    )
    return jnp.maximum(lr, 0.0)

def create_optimizer(learning_rate: float = 1e-4, weight_decay: float = 0.01, 
                    beta1: float = 0.9, beta2: float = 0.999, 
                    warmup_steps: int = 2000, num_train_steps: int = 100000):
    """Create optimizer with TPU-optimized learning rate schedule."""
    num_training_devices = jax.device_count()
    learning_rate_fn = partial(
        create_learning_rate_schedule,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        num_train_steps=num_train_steps,
        weight_decay=weight_decay,
        num_training_devices=num_training_devices
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=learning_rate_fn,
            b1=beta1,
            b2=beta2,
            weight_decay=weight_decay
        )
    )
    return optimizer

# Model Configurations
@dataclass
class ModelArgs:
    """TPU-optimized model arguments."""
    dim: int = 768
    n_layers: int = 32
    n_heads: int = 12
    n_kv_heads: int = 8
    vocab_size: int = 32000
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 1024
    n_experts: int = 8
    expert_dim: int = 4096
    expert_pruning_threshold: float = 0.1
    min_active_experts: int = 4
    dynamic_expert_selection: bool = True
    expert_capacity_factor: float = 1.25
    window_size: int = 512
    global_tokens: int = 64
    attention_dropout: float = 0.1
    dropout_rate: float = 0.1
    expert_dropout: float = 0.1
    use_alibi: bool = True
    use_rope: bool = True
    num_alibi_heads: Optional[int] = None
    use_flash_attention: bool = True
    param_dtype: jnp.dtype = jnp.bfloat16  # Default to bfloat16 for TPU
    use_dualpipe: bool = True
    use_eplb: bool = True
    use_deepgemm: bool = True

@dataclass
class ModelConfig:
    """Enhanced model configuration with TPU optimizations."""
    @classmethod
    def map_config_params(cls, config_dict: Dict) -> Dict:
        mapped_dict = config_dict.copy()
        if 'attention_dropout' in mapped_dict:
            mapped_dict['attention_dropout_prob'] = mapped_dict.pop('attention_dropout')
        if 'dropout' in mapped_dict:
            mapped_dict['hidden_dropout_prob'] = mapped_dict.pop('dropout')
        if 'hidden_size' not in mapped_dict and 'dim' in mapped_dict:
            mapped_dict['hidden_size'] = mapped_dict.pop('dim')
        if 'num_attention_heads' not in mapped_dict and 'num_heads' in mapped_dict:
            mapped_dict['num_attention_heads'] = mapped_dict.pop('num_heads')
        if 'num_layers' not in mapped_dict and 'n_layers' in mapped_dict:
            mapped_dict['num_layers'] = mapped_dict.pop('n_layers')
        if 'intermediate_size' not in mapped_dict and 'intermediate_dim' in mapped_dict:
            mapped_dict['intermediate_size'] = mapped_dict.pop('intermediate_dim')
        mapped_dict.pop('attention_bias', None)
        return {k: v for k, v in mapped_dict.items() if k in cls.__dataclass_fields__}

    vocab_size: int = 32000
    hidden_size: int = 768
    num_layers: int = 32
    num_attention_heads: int = 12
    intermediate_size: int = 11008
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 1024
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = False
    use_flash_attention: bool = True
    use_rope: bool = True
    use_alibi: bool = False
    use_gqa: bool = True
    num_key_value_heads: int = 8
    dtype: str = "bfloat16"  # Default to bfloat16 for TPU
    quantization: Optional[str] = None
    use_dualpipe: bool = True
    use_eplb: bool = True
    use_deepgemm: bool = True
    eplb_window_size: int = 100
    eplb_threshold: float = 0.8

    def __post_init__(self):
        if self.use_gqa:
            assert self.num_attention_heads % self.num_key_value_heads == 0, \
                "num_attention_heads must be divisible by num_key_value_heads for GQA"

# Model Components
class LayerNorm(nn.Module):
    """TPU-optimized Layer Normalization."""
    epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.bfloat16
    scale_init: callable = nn.initializers.ones

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        x = x.astype(jnp.float32)  # Higher precision for stability
        mean = jnp.mean(x, axis=-1, keepdims=True)
        var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
        inv_std = jax.lax.rsqrt(var + self.epsilon)
        scale = self.param('scale', self.scale_init, (x.shape[-1],))
        x = (x - mean) * inv_std * scale
        return x.astype(self.dtype)

class Dense(nn.Module):
    """TPU-optimized Dense layer."""
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.bfloat16
    kernel_init: callable = nn.initializers.normal(stddev=0.02)
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        kernel = self.param('kernel', self.kernel_init, (x.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)
        y = jax.lax.dot_general(x, kernel, (((x.ndim - 1,), (0,)), ((), ())))
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return y

class VishwamAIModel(nn.Module):
    """Main model class with TPU optimizations."""
    config: ModelConfig

    def setup(self):
        dtype = getattr(jnp, self.config.dtype)
        self.embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=dtype
        )
        n_kv_heads = self.config.num_key_value_heads if self.config.use_gqa else self.config.num_attention_heads
        
        # Create TPU-optimized transformer blocks
        self.encoder = [
            TransformerBlock(ModelArgs(
                dim=self.config.hidden_size,
                n_layers=1,
                n_heads=self.config.num_attention_heads,
                n_kv_heads=n_kv_heads,
                vocab_size=self.config.vocab_size,
                multiple_of=256,
                norm_eps=self.config.layer_norm_eps,
                max_batch_size=32,
                max_seq_len=self.config.max_position_embeddings,
                n_experts=4,
                expert_dim=4096,
                expert_capacity_factor=1.25,
                window_size=512,
                global_tokens=64,
                attention_dropout=self.config.attention_dropout_prob,
                dropout_rate=self.config.hidden_dropout_prob,
                expert_dropout=self.config.hidden_dropout_prob,
                param_dtype=dtype,
                use_rope=self.config.use_rope,
                use_flash_attention=self.config.use_flash_attention,
                use_alibi=self.config.use_alibi,
                use_dualpipe=self.config.use_dualpipe,
                use_eplb=self.config.use_eplb,
                use_deepgemm=self.config.use_deepgemm
            )) for _ in range(self.config.num_layers)
        ]
        self.final_layer_norm = LayerNorm(epsilon=self.config.layer_norm_eps, dtype=dtype)
        self.lm_head = Dense(features=self.config.vocab_size, use_bias=False, dtype=dtype)

    def _create_causal_mask(self, seq_len: int) -> jnp.ndarray:
        """Create TPU-optimized causal mask."""
        return jnp.triu(
            jnp.full((seq_len, seq_len), -1e9, dtype=jnp.bfloat16), 
            k=1
        )

    @partial(jax.jit, static_argnums=(0,))
    def __call__(
        self, 
        input_ids: jnp.ndarray, 
        attention_mask: Optional[jnp.ndarray] = None,
        deterministic: bool = True
    ) -> Dict[str, jnp.ndarray]:
        """TPU-optimized forward pass."""
        # Place computation on TPU
        device = jax.devices("tpu")[0] if jax.devices("tpu") else jax.devices("cpu")[0]
        with jax.default_device(device):
            hidden_states = self.embeddings(input_ids)
            
            if attention_mask is None:
                attention_mask = self._create_causal_mask(input_ids.shape[1])
            
            # Process through transformer layers with TPU optimization
            for encoder_layer in self.encoder:
                hidden_states = encoder_layer(hidden_states, attention_mask, deterministic)
            
            hidden_states = self.final_layer_norm(hidden_states)
            logits = self.lm_head(hidden_states)
            
            return {
                'logits': logits,
                'hidden_states': hidden_states
            }

    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[ModelConfig] = None):
        """Load pretrained model with TPU and GCS support."""
        # Handle GCS paths
        if model_path.startswith('gs://'):
            client = storage.Client()
            bucket_name, blob_path = model_path.replace('gs://', '').split('/', 1)
            bucket = client.get_bucket(bucket_name)
            local_path = '/tmp/model'
            os.makedirs(local_path, exist_ok=True)
            
            # Download config and weights
            for blob in bucket.list_blobs(prefix=blob_path):
                if blob.name.endswith(('.safetensors', 'config.json')):
                    local_file = os.path.join(local_path, os.path.basename(blob.name))
                    blob.download_to_filename(local_file)
            model_path = local_path
        elif not os.path.exists(model_path):
            model_path = snapshot_download(repo_id=model_path, allow_patterns=["*.safetensors", "config.json"])
        
        if config is None:
            with open(os.path.join(model_path, "config.json"), 'r') as f:
                config_dict = json.load(f)
                config_dict = ModelConfig.map_config_params(config_dict)
            config = ModelConfig(**config_dict)
        
        model = cls(config)
        params = {}
        
        # Load weights with TPU optimization
        for shard_file in sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")]):
            shard_path = os.path.join(model_path, shard_file)
            shard_params = stf.load_file(shard_path)
            # Convert to bfloat16 for TPU
            for k, v in shard_params.items():
                params[k] = v.astype(jnp.bfloat16)
        
        return model, {'params': params}

    def load_weights(self, model_path: str, reduced_size: bool = False):
        """Load weights with TPU optimization and optional model reduction."""
        if model_path.startswith('gs://'):
            client = storage.Client()
            bucket_name, blob_path = model_path.replace('gs://', '').split('/', 1)
            bucket = client.get_bucket(bucket_name)
            local_path = '/tmp/model_weights'
            os.makedirs(local_path, exist_ok=True)
            
            for blob in bucket.list_blobs(prefix=blob_path):
                if blob.name.endswith('.safetensors'):
                    local_file = os.path.join(local_path, os.path.basename(blob.name))
                    blob.download_to_filename(local_file)
            model_path = local_path
        elif not os.path.exists(model_path):
            model_path = snapshot_download(repo_id=model_path, allow_patterns=["*.safetensors"])
        
        # Adjust model size if needed
        if reduced_size:
            self.config.hidden_size //= 2
            self.config.num_attention_heads //= 2
            self.config.num_key_value_heads = max(1, self.config.num_key_value_heads // 2)
            self.config.intermediate_size //= 2
            self.config.num_layers //= 2
        
        params = {}
        dtype = jnp.bfloat16  # Always use bfloat16 for TPU
        
        # Load and process weights with TPU optimization
        for shard_file in sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")]):
            shard_path = os.path.join(model_path, shard_file)
            with stf.safe_open(shard_path, framework="numpy") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name)
                    if reduced_size and len(tensor.shape) >= 2:
                        new_shape = tuple(s // 2 if i < 2 else s for i, s in enumerate(tensor.shape))
                        tensor = tensor[tuple(slice(0, s) for s in new_shape)]
                    # Place directly on TPU if available
                    device = jax.devices("tpu")[0] if jax.devices("tpu") else jax.devices("cpu")[0]
                    with jax.default_device(device):
                        params[name] = jnp.array(tensor, dtype=dtype)
                    del tensor
                    gc.collect()
        
        self.bind({'params': params})
        return self

# Training and Evaluation Utilities
@partial(jax.jit, static_argnums=(0,))
def train_step(model, state, batch, rng):
    """TPU-optimized training step."""
    def loss_fn(params):
        outputs = model.apply(
            {'params': params},
            batch['input_ids'],
            deterministic=False,
            rngs={'dropout': rng}
        )
        logits = outputs['logits'][:, :-1].astype(jnp.float32)
        labels = batch['labels'][:, 1:]
        loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(state.params)
    # All-reduce gradients across TPU cores
    grads = jax.lax.pmean(grads, axis_name='batch')
    state = state.apply_gradients(grads=grads)
    return state, loss

@partial(jax.jit, static_argnums=(0,))
def eval_step(model, params, batch):
    """TPU-optimized evaluation step."""
    outputs = model.apply(
        {'params': params},
        batch['input_ids'],
        deterministic=True
    )
    logits = outputs['logits'][:, :-1].astype(jnp.float32)
    labels = batch['labels'][:, 1:]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, labels).mean()
    return loss

# Example Usage
if __name__ == "__main__":
    config = ModelConfig()
    model = VishwamAIModel(config)
    
    # Initialize on TPU if available
    device = jax.devices("tpu")[0] if jax.devices("tpu") else jax.devices("cpu")[0]
    with jax.default_device(device):
        rng = jax.random.PRNGKey(0)
        dummy_input = jnp.ones((1, config.max_position_embeddings), dtype=jnp.int32)
        dummy_attention_mask = jnp.ones((config.max_position_embeddings, config.max_position_embeddings), dtype=jnp.int32)
        params = model.init(rng, dummy_input, attention_mask=dummy_attention_mask)['params']
    
    logger.info("Model initialized successfully on TPU!")
