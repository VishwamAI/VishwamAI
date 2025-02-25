import math
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import numpy as np
from einops import rearrange, repeat
import json
import os
import gc
from huggingface_hub import snapshot_download
import safetensors.flax as stf  # Correct import
from omegaconf import OmegaConf
import logging  # Replace import logger with standard logging

# Configure logging
logger = logging.getLogger(__name__)

def create_optimizer(learning_rate: float = 1e-4, weight_decay: float = 0.01, 
                    beta1: float = 0.9, beta2: float = 0.999, 
                    warmup_steps: int = 2000, num_train_steps: int = 100000):
    """
    Creates an optimizer for training the model.
    
    Args:
        learning_rate: Peak learning rate
        weight_decay: Weight decay coefficient
        beta1: First moment coefficient for Adam
        beta2: Second moment coefficient for Adam
        warmup_steps: Number of warmup steps
        num_train_steps: Total number of training steps
    
    Returns:
        An optax optimizer
    """
    decay_scheduler = optax.linear_schedule(
        init_value=0.0,
        end_value=learning_rate,
        transition_steps=warmup_steps
    )
    
    decay_scheduler = optax.join_schedules(
        schedules=[decay_scheduler, optax.linear_schedule(
            init_value=learning_rate,
            end_value=0,
            transition_steps=num_train_steps - warmup_steps
        )],
        boundaries=[warmup_steps]
    )
    
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.adamw(
            learning_rate=decay_scheduler,
            b1=beta1,
            b2=beta2,
            weight_decay=weight_decay
        )
    )
    
    return optimizer

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 32000
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 2048
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
    num_alibi_heads: Optional[int] = None
    use_flash_attention: bool = True
    kv_cache_dtype: jnp.dtype = jnp.int8
    param_dtype: jnp.dtype = jnp.bfloat16
    vision_dim: int = 1024
    use_contrastive_loss: bool = True
    temperature: float = 0.07
    max_image_length: int = 256

@dataclass
class ModelConfig:
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
    hidden_size: int = 4096
    num_layers: int = 32
    num_attention_heads: int = 32
    intermediate_size: int = 11008
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 2048
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
    dtype: str = "bfloat16"
    quantization: Optional[str] = None

    def __post_init__(self):
        if self.use_gqa:
            assert self.num_attention_heads % self.num_key_value_heads == 0, \
                "num_attention_heads must be divisible by num_key_value_heads for GQA"

class ParallelDense(nn.Module):
    features: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32
    precision: Optional[tuple] = None
    kernel_init: callable = nn.initializers.normal(stddev=0.02)
    
    @nn.compact
    def __call__(self, inputs):
        kernel = self.param('kernel', self.kernel_init, (inputs.shape[-1], self.features))
        kernel = jnp.asarray(kernel, self.dtype)
        y = jnp.dot(inputs, kernel, precision=self.precision)
        if self.use_bias:
            bias = self.param('bias', nn.initializers.zeros, (self.features,))
            bias = jnp.asarray(bias, self.dtype)
            y = y + bias
        return y

class RMSNorm(nn.Module):
    epsilon: float = 1e-6
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, x):
        variance = jnp.mean(jnp.square(x), axis=-1, keepdims=True)
        x = x * jax.lax.rsqrt(variance + self.epsilon)
        scale = self.param('scale', nn.initializers.ones, (x.shape[-1],))
        return x * jnp.asarray(scale, self.dtype)

def rotary_embedding(x: jnp.ndarray, freqs: jnp.ndarray) -> jnp.ndarray:
    sin, cos = freqs
    sin = repeat(sin, '... d -> ... (d 2)')
    cos = repeat(cos, '... d -> ... (d 2)')
    x1, x2 = rearrange(x, '... (d r) -> ... d r', r=2).unbind(-1)
    rotated = jnp.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], axis=-1)
    return rearrange(rotated, '... d r -> ... (d r)')

def precompute_freqs(dim: int, max_seq_len: int, base: int = 10000) -> Tuple[jnp.ndarray, jnp.ndarray]:
    freqs = 1.0 / (base ** (jnp.arange(0, dim, 2)[: (dim // 2)].astype(jnp.float32) / dim))
    t = jnp.arange(max_seq_len)
    freqs = jnp.outer(t, freqs)
    sin, cos = jnp.sin(freqs), jnp.cos(freqs)
    return sin, cos

class ParallelMLP(nn.Module):
    config: ModelArgs
    
    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        dim = self.config.dim
        hidden_dim = int(self.config.ffn_dim_multiplier * dim if self.config.ffn_dim_multiplier else dim * 4)
        
        x = RMSNorm(dtype=x.dtype)(x)
        gate = ParallelDense(hidden_dim, dtype=x.dtype)(x)
        up = ParallelDense(hidden_dim, dtype=x.dtype)(x)
        gate = nn.silu(gate)
        intermediate = gate * up
        output = ParallelDense(dim, dtype=x.dtype)(intermediate)
        return nn.Dropout(rate=self.config.dropout_rate)(output, deterministic=deterministic)

class MoELayer(nn.Module):
    config: ModelArgs
    
    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        batch_size, seq_len, dim = x.shape
        num_experts = self.config.n_experts
        
        expert_fn = ParallelMLP(self.config)
        router_weights = self.param('router_weights', nn.initializers.normal(stddev=0.02), (dim, num_experts))
        
        router_logits = jax.lax.stop_gradient(jnp.einsum('bsd,de->bse', x, router_weights))
        routing_probs = nn.softmax(router_logits, axis=-1)
        
        top2_weights, top2_indices = jax.lax.top_k(routing_probs, k=2)
        top2_weights = top2_weights / jnp.sum(top2_weights, axis=-1, keepdims=True)
        
        expert_inputs = []
        expert_indices = []
        
        for expert_idx in range(num_experts):
            expert_mask = (top2_indices == expert_idx).any(axis=-1)
            if jnp.any(expert_mask):
                mask_weights = jnp.where(top2_indices == expert_idx, top2_weights, 0).max(axis=-1)
                expert_inputs.append(x[expert_mask] * mask_weights[expert_mask, None])
                expert_indices.append(jnp.where(expert_mask)[0])
        
        expert_outputs = []
        if expert_inputs:
            batched_expert_input = jnp.concatenate(expert_inputs, axis=0)
            batched_expert_output = expert_fn(batched_expert_input, deterministic)
            offset = 0
            for expert_input in expert_inputs:
                expert_len = len(expert_input)
                expert_outputs.append((expert_indices[offset:offset + expert_len], batched_expert_output[offset:offset + expert_len]))
                offset += expert_len
        
        final_output = jnp.zeros_like(x)
        for indices, outputs in expert_outputs:
            final_output = final_output.at[indices].add(outputs)
        
        expert_usage = jnp.mean(routing_probs, axis=(0, 1))
        load_balancing_loss = -jnp.sum(expert_usage * jnp.log(expert_usage + 1e-6))
        
        return final_output, load_balancing_loss

def create_alibi_slopes(num_heads: int) -> jnp.ndarray:
    closest_power_of_2 = 2 ** jnp.floor(jnp.log2(num_heads))
    base = jnp.array([2 ** (-(2 ** -(jnp.log2(closest_power_of_2) - 3)))], dtype=jnp.float32)
    powers = jnp.arange(1, 1 + num_heads, dtype=jnp.float32)
    slopes = jnp.power(base, powers)
    return slopes

class MultiheadAttention(nn.Module):
    config: ModelArgs
    
    def create_sliding_window_mask(self, seq_len: int) -> jnp.ndarray:
        window_size = self.config.window_size
        global_tokens = self.config.global_tokens
        mask = jnp.full((seq_len, seq_len), -1e9)
        for i in range(seq_len):
            window_start = max(0, i - window_size // 2)
            window_end = min(seq_len, i + window_size // 2 + 1)
            mask = mask.at[i, window_start:window_end].set(0.0)
        mask = mask.at[:global_tokens, :].set(0.0)
        mask = mask.at[:, :global_tokens].set(0.0)
        return mask
    
    def setup(self):
        if self.config.use_alibi:
            num_alibi_heads = self.config.num_alibi_heads or self.config.n_heads
            self.alibi_slopes = create_alibi_slopes(num_alibi_heads)
    
    def compute_alibi_attention(self, qk: jnp.ndarray) -> jnp.ndarray:
        seq_len = qk.shape[-1]
        positions = jnp.arange(seq_len)
        distance = positions[:, None] - positions[None, :]
        distance = -jnp.abs(distance).astype(jnp.float32)
        distance = distance[None, :, :]
        alibi_bias = self.alibi_slopes[:, None, None] * distance
        return qk + alibi_bias

    @nn.compact
    def __call__(self, x, mask=None, deterministic: bool = True):
        batch_size, seq_len, dim = x.shape
        num_heads = self.config.n_heads
        num_kv_heads = self.config.n_kv_heads
        head_dim = dim // num_heads
        
        token_complexity = jnp.sum(jnp.abs(x), axis=-1, keepdims=True)
        complexity_weights = nn.sigmoid(token_complexity)
        
        q = nn.remat(ParallelDense)(dim, use_bias=False, dtype=x.dtype)(x * complexity_weights)
        kv = nn.remat(ParallelDense)(2 * dim * (num_kv_heads / num_heads), use_bias=False, dtype=x.dtype)(x)
        k, v = jnp.split(kv, 2, axis=-1)
        
        q = rearrange(q, 'b s (h d) -> b h s d', h=num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=num_kv_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=num_kv_heads)
        
        repeats = num_heads // num_kv_heads
        k = repeat(k, 'b h s d -> b (h r) s d', r=repeats)
        v = repeat(v, 'b h s d -> b (h r) s d', r=repeats)
        
        if mask is None:
            mask = self.create_sliding_window_mask(seq_len)
        else:
            window_mask = self.create_sliding_window_mask(seq_len)
            mask = jnp.minimum(mask, window_mask)
        
        scale = 1.0 / jnp.sqrt(head_dim)
        qk = jnp.einsum('bhid,bhjd->bhij', q, k) * scale
        
        if self.config.use_alibi:
            qk = self.compute_alibi_attention(qk)
        
        if self.config.use_flash_attention:
            qk = jnp.where(mask, qk, -1e9)
            attention = nn.softmax(qk, axis=-1)
        else:
            qk = jnp.where(mask, qk, -1e9)
            attention = nn.softmax(qk, axis=-1)
        
        if not deterministic:
            attention = nn.Dropout(rate=self.config.attention_dropout)(attention, deterministic=False)
        
        if not deterministic:
            v = jnp.asarray(v, self.config.kv_cache_dtype)
            
        output = jnp.einsum('bhij,bhjd->bhid', attention, v)
        output = rearrange(output, 'b h s d -> b s (h d)')
        
        output = nn.remat(ParallelDense)(dim, dtype=x.dtype)(output)
        output = nn.Dropout(rate=self.config.dropout_rate)(output, deterministic=deterministic)
        
        return output

class TransformerBlock(nn.Module):
    config: ModelArgs
    
    @nn.compact
    def __call__(self, x, mask=None, deterministic: bool = True):
        attn_norm = RMSNorm(dtype=x.dtype)(x)
        attn_output = MultiheadAttention(self.config)(attn_norm, mask, deterministic)
        x = x + attn_output
        
        ff_norm = RMSNorm(dtype=x.dtype)(x)
        if self.config.n_experts > 0:
            ff_output, load_balance_loss = MoELayer(self.config)(ff_norm, deterministic)
            self.sow('intermediates', 'load_balance_loss', load_balance_loss)
        else:
            ff_output = ParallelMLP(self.config)(ff_norm, deterministic)
        
        x = x + ff_output
        return x

class ParallelEmbedding(nn.Module):
    args: ModelArgs
    dtype: jnp.dtype = jnp.float32
    param_dtype: jnp.dtype = jnp.float32
    
    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        embedding = self.param('embedding', nn.initializers.normal(stddev=0.02), (self.args.vocab_size, self.args.dim), self.param_dtype)
        if jnp.any((x < 0) | (x >= self.args.vocab_size)):
            raise ValueError("Input ids must be in range [0, vocab_size)")
        embedded = jnp.take(embedding, x, axis=0)
        return jnp.asarray(embedded, self.dtype)

def download_partial_model(model_path: str, num_shards: int = 15):
    """Download only specified number of model shards"""
    patterns = [f"model-{i+1:05d}-of-00252.safetensors" for i in range(num_shards)]
    patterns.extend(["config.json", "tokenizer.model"])
    try:
        local_path = snapshot_download(
            repo_id=model_path,
            allow_patterns=patterns,
            local_files_only=False,
            resume_download=True
        )
        print(f"Successfully downloaded {num_shards} model shards to {local_path}")
        return local_path
    except Exception as e:
        raise ValueError(f"Error downloading model shards: {str(e)}")

class VishwamAIModel(nn.Module):
    config: ModelConfig

    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[ModelConfig] = None, rename_params: bool = True):
        if not os.path.exists(model_path):
            try:
                model_path = snapshot_download(
                    repo_id=model_path,
                    allow_patterns=["*.safetensors", "config.json", "tokenizer.model"]
                )
            except Exception as e:
                raise ValueError(f"Error downloading model from {model_path}: {str(e)}")
        
        if config is None:
            config_path = os.path.join(model_path, "config.json")
            if not os.path.exists(config_path):
                raise ValueError(f"Config not found at {config_path}")
            with open(config_path) as f:
                config_dict = json.load(f)
            if rename_params:
                config_dict = ModelConfig.map_config_params(config_dict)
            config = ModelConfig(**config_dict)
        
        model = cls(config)
        params = {}
        shard_files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
        
        if not shard_files:
            raise ValueError(f"No .safetensors files found in {model_path}")
        
        for shard_file in shard_files:
            shard_path = os.path.join(model_path, shard_file)
            try:
                shard_params = stf.load_file(shard_path)
                params.update(shard_params)
            except Exception as e:
                raise ValueError(f"Error loading weights from {shard_path}: {str(e)}")
        
        variables = {'params': params}
        return model.bind(variables)

    def load_weights(self, model_path: str, reduced_size: bool = True):
        """Load pretrained weights with option to reduce model size for memory efficiency."""
        import safetensors  # P3432
        print(f"Debug: Using safetensors.flax as stf: {stf}")  # Debug to confirm stf is imported
        if not os.path.exists(model_path):
            try:
                model_path = snapshot_download(
                    repo_id=model_path,
                    allow_patterns=["*.safetensors"]
                )
            except Exception as e:
                raise ValueError(f"Error downloading model from {model_path}: {str(e)}")

        if reduced_size:
            print("Loading reduced size model for memory constraints...")
            self.config.hidden_size = self.config.hidden_size // 2
            self.config.num_attention_heads = self.config.num_attention_heads // 2
            self.config.num_key_value_heads = max(1, self.config.num_key_value_heads // 2)
            self.config.intermediate_size = self.config.intermediate_size // 2
            self.config.num_layers = self.config.num_layers // 2
        
        params = {}
        shard_files = sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")])
        
        if not shard_files:
            raise ValueError(f"No .safetensors files found in {model_path}")
        
        for shard_file in shard_files:
            shard_path = os.path.join(model_path, shard_file)
            try:
                print(f"Loading {shard_file}...")
                with stf.safe_open(shard_path, framework="numpy") as f:
                    for name in f.keys():
                        try:
                            tensor = f.get_tensor(name).astype(np.float16)
                            if reduced_size and len(tensor.shape) >= 2:
                                new_shape = tuple(s // 2 if i < 2 else s for i, s in enumerate(tensor.shape))
                                tensor = tensor[tuple(slice(0, s) for s in new_shape)]
                            with jax.default_device(jax.devices("cpu")[0]):
                                params[name] = jnp.array(tensor)
                            del tensor
                            gc.collect()
                        except Exception as e:
                            print(f"Warning: Could not load {name}, skipping... Error: {str(e)}")
                            continue
            except Exception as e:
                params.clear()
                gc.collect()
                jax.clear_backends()
                if "RESOURCE_EXHAUSTED" in str(e):
                    raise ValueError(
                        "Failed to load model due to memory constraints.\n"
                        "Options:\n1. Use Colab Pro/Pro+\n2. Try a smaller model\n3. Use CPU offloading\n"
                        f"Error: {str(e)}"
                    )
                raise ValueError(f"Error loading weights from {shard_path}: {str(e)}")
        
        gc.collect()
        jax.clear_backends()

        self.bind({'params': params})
        return self

    def setup(self):
        self.embeddings = self._create_embeddings()
        self.encoder = self._create_encoder()
        self.decoder = self._create_decoder()
        self.final_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=jnp.dtype(self.config.dtype))
        
        # Add ToT integration components
        self.use_tot = getattr(self.config, 'use_tot', False)
        if self.use_tot:
            from vishwamai.tot import TreeOfThoughts
            from vishwamai.transformer import VishwamAIModel as VisionTransformer10B
            from vishwamai.integration import ToTIntegrationLayer, MultiLevelToTAttention
            
            # Create dummy vision transformer for ToT
            self.vision_transformer = VisionTransformer10B(self.config)
            
            # Create ToT model
            self.tot_model = TreeOfThoughts(
                transformer=self.vision_transformer,
                max_thoughts=getattr(self.config, 'tot_max_thoughts', 5),
                max_depth=getattr(self.config, 'tot_max_depth', 3),
                beam_width=getattr(self.config, 'tot_beam_width', 8)
            )
            
            # Create integration components
            self.tot_integration = ToTIntegrationLayer(self.config)
            self.tot_mla = MultiLevelToTAttention(
                hidden_size=self.config.hidden_size,
                num_heads=min(8, self.config.num_attention_heads)
            )
            
        # Add MoD components
        self.use_mod = getattr(self.config, 'use_mod', False)
        if self.use_mod:
            from vishwamai.integration import MixtureDensityNetwork
            self.mod_layer = MixtureDensityNetwork(
                hidden_size=self.config.hidden_size,
                num_mixtures=getattr(self.config, 'mod_num_mixtures', 5)
            )
    
    def _create_embeddings(self):
        return nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=jnp.dtype(self.config.dtype)
        )
    
    def _create_encoder(self):
        return [TransformerBlock(ModelArgs(
            dim=self.config.hidden_size,
            n_layers=1,
            n_heads=self.config.num_attention_heads,
            n_kv_heads=self.config.num_key_value_heads if self.config.use_gqa else self.config.num_attention_heads,
            vocab_size=self.config.vocab_size
        )) for _ in range(self.config.num_layers)]
    
    def _create_decoder(self):
        return [TransformerBlock(ModelArgs(
            dim=self.config.hidden_size,
            n_layers=1,
            n_heads=self.config.num_attention_heads,
            n_kv_heads=self.config.num_key_value_heads if self.config.use_gqa else self.config.num_attention_heads,
            vocab_size=self.config.vocab_size
        )) for _ in range(self.config.num_layers)]
    
    def __call__(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None, 
                position_ids: Optional[jnp.ndarray] = None, deterministic: bool = True,
                use_tot: bool = None, tot_rng_key: Optional[jnp.ndarray] = None) -> Dict[str, jnp.ndarray]:
        """
        Enhanced forward pass with Tree of Thoughts integration.
        
        Args:
            input_ids: Input token IDs
            attention_mask: Attention mask
            position_ids: Position IDs
            deterministic: Whether to use deterministic mode
            use_tot: Whether to use Tree of Thoughts (overrides config)
            tot_rng_key: Random key for ToT search
        """
        hidden_states = self.embeddings(input_ids)
        
        if attention_mask is None:
            attention_mask = self._create_causal_mask(input_ids.shape[1])
        
        encoder_outputs = []
        for encoder_layer in self.encoder:
            hidden_states = encoder_layer(hidden_states, attention_mask, deterministic)
            encoder_outputs.append(hidden_states)
        
        # Apply MoD if enabled
        mod_weights = None
        if self.use_mod:
            hidden_states, mod_weights = self.mod_layer(hidden_states, deterministic)
        
        decoder_outputs = []
        for decoder_layer in self.decoder:
            hidden_states = decoder_layer(hidden_states, attention_mask, deterministic)
            decoder_outputs.append(hidden_states)
        
        hidden_states = self.final_layer_norm(hidden_states)
        
        # Apply Tree of Thoughts if enabled
        tot_outputs = None
        if (use_tot is True) or (use_tot is None and self.use_tot):
            try:
                # Generate thoughts
                if tot_rng_key is None:
                    tot_rng_key = jax.random.PRNGKey(0)
                
                tot_thought = self.tot_model(hidden_states, tot_rng_key)
                
                # Collect thought features
                thought_features = []
                current_thought = tot_thought
                while current_thought:
                    thought_features.append(current_thought.embeddings)
                    if current_thought.children:
                        current_thought = max(current_thought.children, key=lambda t: t.score)
                    else:
                        break
                
                # Apply multi-level attention if we have thoughts
                if thought_features:
                    tot_features = jnp.stack(thought_features)
                    enhanced_features, attn_weights = self.tot_mla(
                        hidden_states,
                        tot_features,
                        thought_features,
                        deterministic
                    )
                    
                    # Integrate features
                    integrated_features, integration_info = self.tot_integration(
                        hidden_states,
                        enhanced_features,
                        deterministic
                    )
                    
                    # Use integrated features
                    hidden_states = integrated_features
                    
                    tot_outputs = {
                        'thought': tot_thought,
                        'attention_weights': attn_weights,
                        'integration_info': integration_info
                    }
            
            except Exception as e:
                logger.warning(f"ToT integration failed: {e}")
        
        outputs = {
            'last_hidden_state': hidden_states,
            'encoder_outputs': encoder_outputs,
            'decoder_outputs': decoder_outputs,
        }
        
        if tot_outputs is not None:
            outputs['tot_outputs'] = tot_outputs
            
        if mod_weights is not None:
            outputs['mod_weights'] = mod_weights
            
        return outputs

# Placeholder classes for missing dependencies
class VishwamAITokenizer:
    def __init__(self, vocab_size: int, model_prefix: str):
        self.vocab_size = vocab_size
        self.model_prefix = model_prefix
        print(f"Initialized dummy tokenizer with vocab_size={vocab_size}, prefix={model_prefix}")

class VishwamaiShaalaTrainer:
    def __init__(self, teacher_model, student_model, cfg):
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.cfg = cfg
        print("Initialized dummy trainer")

# Main execution
if __name__ == "__main__":
    # Load distillation configuration
    distillation_config_path = os.path.join("vishwamai", "configs", "training", "perplexity_r1_distillation.yaml")
    if not os.path.exists(distillation_config_path):
        distillation_config = OmegaConf.create({
            'distillation': {
                'teacher_model': {
                    'path': "perplexity-ai/r1-1776",
                    'config': {
                        'hidden_size': 7168,
                        'intermediate_size': 18432,
                        'num_attention_heads': 128,
                        'num_layers': 61,
                        'num_key_value_heads': 128,
                        'vocab_size': 129280,
                        'max_position_embeddings': 163840
                    }
                },
                'student_model': {
                    'path': "model-00001-to-00015-of-00252.safetensors",
                    'config': {
                        'hidden_size': 2048,
                        'intermediate_size': 8192,
                        'num_attention_heads': 32,
                        'num_layers': 24,
                        'num_key_value_heads': 32,
                        'vocab_size': 129280,
                        'max_position_embeddings': 163840
                    }
                }
            }
        })
        print(f"Warning: Config file not found at {distillation_config_path}. Using default config.")
    else:
        distillation_config = OmegaConf.load(distillation_config_path)

    # Download partial teacher model
    teacher_path = download_partial_model(
        distillation_config['distillation']['teacher_model']['path'],
        num_shards=5
    )

    # Initialize teacher model
    teacher_config = distillation_config['distillation']['teacher_model']['config']
    student_config = distillation_config['distillation']['student_model']['config']

    teacher_model = VishwamAIModel(ModelConfig(**teacher_config))
    teacher_model.load_weights(teacher_path, reduced_size=True)

    # Initialize student model
    student_model = VishwamAIModel(ModelConfig(**student_config))

    # Initialize tokenizer (placeholder)
    tokenizer = VishwamAITokenizer(
        vocab_size=teacher_config["vocab_size"],
        model_prefix="vishwamai"
    )

    # Initialize trainer (placeholder)
    trainer = VishwamaiShaalaTrainer(
        teacher_model=teacher_model,
        student_model=student_model,
        cfg=distillation_config
    )

    print("Setup completed successfully!")
