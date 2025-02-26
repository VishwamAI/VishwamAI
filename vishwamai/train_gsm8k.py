"""
Enhanced GSM8K training module with deep ToT integration, optimized for TPU v2-8.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from datasets import load_dataset
import os
import logging
from safetensors.flax import save_file
from typing import Dict, Iterator, Optional
import random
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict, Union
import flax.linen as nn
from einops import rearrange, repeat
import gc
from huggingface_hub import snapshot_download
import safetensors.flax as stf
from omegaconf import OmegaConf
import json
logger = logging.getLogger(__name__)

from vishwamai.training import train, create_train_state
from vishwamai.tokenizer import VishwamAITokenizer
from vishwamai.error_correction import ErrorCorrectionTrainer
from vishwamai.tot import TreeOfThoughts, Thought

@dataclass
class ModelArgs:
    dim: int = 384
    n_layers: int = 8
    n_heads: int = 12
    n_kv_heads: int = 8
    vocab_size: int = 32000
    multiple_of: int = 256
    ffn_dim_multiplier: Optional[float] = None
    norm_eps: float = 1e-5
    max_batch_size: int = 32
    max_seq_len: int = 512
    n_experts: int = 4
    expert_dim: int = 4096
    expert_pruning_threshold: float = 0.1
    min_active_experts: int = 4
    dynamic_expert_selection: bool = True
    expert_capacity_factor: float = 1.25
    window_size: int = 256
    global_tokens: int = 32
    attention_dropout: float = 0.1
    dropout_rate: float = 0.1
    expert_dropout: float = 0.1
    use_alibi: bool = True
    use_rope: bool = True
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
    hidden_size: int = 384
    num_layers: int = 8
    num_attention_heads: int = 12
    intermediate_size: int = 11008
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 512
    initializer_range: float = 0.02
    layer_norm_eps: float = 1e-5
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2
    tie_word_embeddings: bool = True
    gradient_checkpointing: bool = True
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

def precompute_freqs(head_dim: int, max_seq_len: int, base: float = 10000.0, dtype: jnp.dtype = jnp.float32) -> Tuple[jnp.ndarray, jnp.ndarray]:
    half_dim = head_dim // 2
    freqs = 1.0 / (base ** (jnp.arange(0, half_dim, dtype=dtype) / half_dim))
    t = jnp.arange(max_seq_len, dtype=dtype)
    freqs = jnp.outer(t, freqs)
    sin = jnp.sin(freqs)
    cos = jnp.cos(freqs)
    sin = jnp.expand_dims(jnp.expand_dims(sin, 0), 0)
    cos = jnp.expand_dims(jnp.expand_dims(cos, 0), 0)
    return sin.astype(dtype), cos.astype(dtype)

def rotary_embedding(x: jnp.ndarray, sin: jnp.ndarray, cos: jnp.ndarray, use_local_repeat: bool = False) -> jnp.ndarray:
    batch, heads, seq_len, head_dim = x.shape
    half_dim = head_dim // 2
    sin = sin[:, :, :seq_len, :half_dim]
    cos = cos[:, :, :seq_len, :half_dim]
    sin = jnp.broadcast_to(sin, (batch, heads, seq_len, half_dim))
    cos = jnp.broadcast_to(cos, (batch, heads, seq_len, half_dim))
    x1, x2 = jnp.split(x, 2, axis=-1)
    x_rot = x1 * cos - x2 * sin
    x_pass = x1 * sin + x2 * cos
    return jnp.concatenate([x_rot, x_pass], axis=-1)

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
        top_k_weights, top_k_indices = jax.lax.top_k(routing_probs, k=1)
        top_k_weights = top_k_weights / jnp.sum(top_k_weights, axis=-1, keepdims=True)
        
        all_indices = []
        all_outputs = []
        for expert_idx in range(num_experts):
            expert_mask = (top_k_indices == expert_idx).any(axis=-1)
            if jnp.any(expert_mask):
                mask_weights = jnp.where(top_k_indices == expert_idx, top_k_weights, 0).max(axis=-1)
                expert_input = x[expert_mask] * mask_weights[expert_mask, None]
                expert_output = expert_fn(expert_input, deterministic)
                indices = jnp.where(expert_mask)[0]
                all_indices.append(indices)
                all_outputs.append(expert_output)
        
        final_output = jnp.zeros_like(x)
        if all_indices:
            flat_indices = jnp.concatenate(all_indices, axis=0)
            flat_outputs = jnp.concatenate(all_outputs, axis=0)
            
            # For each index, add its output to the corresponding position in final_output
            for idx, output in zip(flat_indices, flat_outputs):
                final_output = final_output.at[idx].add(output)
        
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
        num_kv_heads = self.config.n_kv_heads if hasattr(self.config, 'n_kv_heads') else num_heads
        head_dim = dim // num_heads
        assert num_heads % num_kv_heads == 0, "num_heads must be divisible by num_kv_heads for GQA"
        
        token_complexity = jnp.sum(jnp.abs(x), axis=-1, keepdims=True)
        complexity_weights = nn.sigmoid(token_complexity)
        
        kv_dim = head_dim * num_kv_heads
        kv_proj_dim = 2 * kv_dim
        
        q = nn.remat(ParallelDense)(dim, use_bias=False, dtype=x.dtype)(x * complexity_weights)
        kv = nn.remat(ParallelDense)(kv_proj_dim, use_bias=False, dtype=x.dtype)(x)
        
        k, v = jnp.split(kv, 2, axis=-1)
        
        q = rearrange(q, 'b s (h d) -> b h s d', h=num_heads)
        k = rearrange(k, 'b s (h d) -> b h s d', h=num_kv_heads)
        v = rearrange(v, 'b s (h d) -> b h s d', h=num_kv_heads)
        
        if self.config.use_rope:
            compute_dtype = x.dtype
            sin, cos = precompute_freqs(head_dim, seq_len, dtype=compute_dtype)
            q = rotary_embedding(q, sin, cos, use_local_repeat=False)
            k = rotary_embedding(k, sin, cos, use_local_repeat=True)
        
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

class VishwamAIModel(nn.Module):
    config: ModelConfig

    def setup(self):
        self.embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=jnp.dtype(self.config.dtype)
        )
        self.encoder = [TransformerBlock(ModelArgs(
            dim=self.config.hidden_size,
            n_layers=1,
            n_heads=self.config.num_attention_heads,
            n_kv_heads=self.config.num_key_value_heads if self.config.use_gqa else self.config.num_attention_heads,
            vocab_size=self.config.vocab_size,
            use_rope=self.config.use_rope,
            dropout_rate=self.config.hidden_dropout_prob,
            attention_dropout=self.config.attention_dropout_prob,
            n_experts=4,
            expert_dim=4096,
            expert_capacity_factor=1.25,
            max_seq_len=self.config.max_position_embeddings,
            window_size=512,
            global_tokens=64,
            max_batch_size=32,
            use_gqa=self.config.use_gqa,
            use_flash_attention=self.config.use_flash_attention,
            use_alibi=self.config.use_alibi
        )) for _ in range(self.config.num_layers)]
        self.final_layer_norm = nn.LayerNorm(epsilon=self.config.layer_norm_eps, dtype=jnp.dtype(self.config.dtype))
        self.lm_head = ParallelDense(features=self.config.vocab_size, use_bias=False, dtype=jnp.dtype(self.config.dtype))

    def _create_causal_mask(self, seq_len: int) -> jnp.ndarray:
        mask = jnp.tril(jnp.ones((seq_len, seq_len)))
        return jnp.where(mask, 0.0, -1e9)

    def __call__(self, input_ids: jnp.ndarray, attention_mask: Optional[jnp.ndarray] = None, 
                 deterministic: bool = True, use_tot: bool = False) -> Dict[str, jnp.ndarray]:
        hidden_states = self.embeddings(input_ids)
        
        if attention_mask is None:
            attention_mask = self._create_causal_mask(input_ids.shape[1])
        
        encoder_outputs = []
        for encoder_layer in self.encoder:
            hidden_states = encoder_layer(hidden_states, attention_mask, deterministic)
            encoder_outputs.append(hidden_states)
        
        hidden_states = self.final_layer_norm(hidden_states)
        logits = self.lm_head(hidden_states)
        
        outputs = {
            'logits': logits,
            'hidden_states': hidden_states,
            'encoder_outputs': encoder_outputs
        }
        
        if use_tot and hasattr(self, 'tot_model'):
            tot_outputs = self.tot_model.generate(input_ids, attention_mask)
            outputs['tot_outputs'] = tot_outputs
            
        return outputs

    @classmethod
    def from_pretrained(cls, model_path: str, config: Optional[ModelConfig] = None):
        if not os.path.exists(model_path):
            model_path = snapshot_download(repo_id=model_path, allow_patterns=["*.safetensors", "config.json"])
        if config is None:
            with open(os.path.join(model_path, "config.json"), 'r') as f:
                config_dict = json.load(f)
                config_dict = ModelConfig.map_config_params(config_dict)
            config = ModelConfig(**config_dict)
        
        model = cls(config)
        params = {}
        for shard_file in sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")]):
            shard_path = os.path.join(model_path, shard_file)
            params.update(stf.load_file(shard_path))
        return model, {'params': params}

    def load_weights(self, model_path: str, reduced_size: bool = False):
        if not os.path.exists(model_path):
            model_path = snapshot_download(repo_id=model_path, allow_patterns=["*.safetensors"])
        if reduced_size:
            self.config.hidden_size //= 2
            self.config.num_attention_heads //= 2
            self.config.num_key_value_heads = max(1, self.config.num_key_value_heads // 2)
            self.config.intermediate_size //= 2
            self.config.num_layers //= 2
        
        params = {}
        dtype = getattr(jnp, self.config.dtype)
        for shard_file in sorted([f for f in os.listdir(model_path) if f.endswith(".safetensors")]):
            shard_path = os.path.join(model_path, shard_file)
            with stf.safe_open(shard_path, framework="numpy") as f:
                for name in f.keys():
                    tensor = f.get_tensor(name).astype(np.float32)
                    if reduced_size and len(tensor.shape) >= 2:
                        new_shape = tuple(s // 2 if i < 2 else s for i, s in enumerate(tensor.shape))
                        tensor = tensor[tuple(slice(0, s) for s in new_shape)]
                    with jax.default_device(jax.devices("cpu")[0]):
                        params[name] = jnp.array(tensor, dtype=dtype)
                    del tensor
                    gc.collect()
        self.bind({'params': params})
        return self

class GSM8KProcessor:
    def __init__(self, tokenizer: VishwamAITokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.data.max_seq_length
    
    def validate_dataset_features(self, dataset):
        required = ['question', 'answer']
        if not all(feat in dataset.features for feat in required):
            raise ValueError(f"Dataset missing required features: {dataset.features}")
        if not all(dataset.features[feat].dtype == 'string' for feat in required):
            raise ValueError(f"Features must be strings: {dataset.features}")

    def evaluate_step_accuracy(self, prediction: str, target: str) -> Dict[str, float]:
        pred_steps = [s.strip() for s in prediction.split('\n') if s.strip().startswith('Step:')]
        target_steps = [s.strip() for s in target.split('\n') if s.strip().startswith('Step:')]
        
        correct_steps = sum(1 for p, t in zip(pred_steps, target_steps) if p == t)
        total_steps = max(len(pred_steps), len(target_steps))
        
        pred_answer = prediction.split('####')[-1].strip() if '####' in prediction else pred_steps[-1].split()[-1] if pred_steps else ""
        target_answer = target.split('####')[-1].strip() if '####' in target else target_steps[-1].split()[-1] if target_steps else ""
        
        exact_match = pred_answer == target_answer and len(pred_steps) == len(target_steps) and all(p == t for p, t in zip(pred_steps, target_steps))
        
        return {
            'step_accuracy': correct_steps / total_steps if total_steps > 0 else 0.0,
            'exact_match': 1.0 if exact_match else 0.0,
            'answer_match': 1.0 if pred_answer == target_answer else 0.0
        }
    
    def tokenize_function(self, examples):
        try:
            questions = examples['question']
            answers = examples['answer']
            
            if len(questions) != len(answers):
                raise ValueError(f"Mismatched lengths: questions={len(questions)}, answers={len(answers)}")
            
            formatted_texts = []
            for q, a in zip(questions, answers):
                formatted_text = f"Question: {q}\nLet's solve this step by step:\n"
                solution_parts = a.split('####')
                steps = solution_parts[0].strip().split('\n')
                final_answer = solution_parts[1].strip() if len(solution_parts) > 1 else steps[-1].strip()
                
                formatted_steps = [f"Step: {step.strip()}" for step in steps if step.strip()]
                formatted_text += f"{chr(10).join(formatted_steps)}\nTherefore, the final answer is: {final_answer}\n####\n{final_answer}"
                formatted_texts.append(formatted_text)
            
            tokenized = self.tokenizer(
                formatted_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            tokenized["error_weights"] = np.ones_like(tokenized["input_ids"], dtype=np.float32)
            answer_pos = formatted_texts[0].find("####")
            if answer_pos != -1:
                token_pos = len(self.tokenizer.encode(formatted_texts[0][:answer_pos]))
                tokenized["error_weights"][:, token_pos:] = 2.0
            
            return tokenized
        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            raise
    
    def prepare_dataset(self, dataset):
        self.validate_dataset_features(dataset)
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=32,
            num_proc=1,
            remove_columns=dataset.column_names,
            desc="Tokenizing GSM8K dataset"
        )
        return tokenized_dataset
    
    def collate_fn(self, examples):
        batch = {
            "input_ids": np.array([ex["input_ids"] for ex in examples]),
            "attention_mask": np.array([ex["attention_mask"] for ex in examples]),
            "labels": np.array([ex["labels"] for ex in examples]),
            "error_weights": np.array([ex["error_weights"] for ex in examples])
        }
        return batch

def create_gsm8k_dataloader(config, tokenizer: VishwamAITokenizer, split="train") -> Iterator:
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    processor = GSM8KProcessor(tokenizer, config)
    processed_dataset = processor.prepare_dataset(dataset)
    
    def data_iterator():
        epoch = 0
        while True:
            indices = list(range(len(processed_dataset)))
            if config.training.get('use_curriculum', True) and epoch < config.training.max_steps // config.training.eval_every:
                indices.sort(key=lambda i: len(processed_dataset[i]['input_ids']))
            random.shuffle(indices)
            
            for i in range(0, len(indices), config.data.batch_size):
                batch_indices = indices[i:i + config.data.batch_size]
                examples = [processed_dataset[idx] for idx in batch_indices]
                yield processor.collate_fn(examples)
            epoch += 1
            logger.info(f"Epoch {epoch} completed")
    
    return data_iterator()

def save_model_safetensors(params: Dict, metrics: Dict, save_path: str):
    numpy_params = jax.tree_map(np.asarray, params)
    metadata = {f"metric_{k}": str(v) for k, v in metrics.items()}
    metadata["timestamp"] = str(np.datetime64('now'))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_file(numpy_params, save_path, metadata=metadata)
    logger.info(f"Saved model to {save_path} with metrics: {metrics}")

def setup_tpu_cluster():
    devices = jax.devices()
    logger.info(f"Available devices: {devices}")
    device_count = len(devices)
    device_mesh = np.array(devices).reshape((device_count // 2, 2))
    mesh = Mesh(device_mesh, ('data', 'model'))
    sharding = NamedSharding(mesh, P('data', 'model'))
    return mesh, sharding

def main(config_path: str = "vishwamai/configs/training/gsm8k.yaml"):
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    config = OmegaConf.load(config_path)
    model_params = {k: v for k, v in config.model.items() if k in ModelConfig.__dataclass_fields__}
    model_config = ModelConfig(**model_params)
    model = VishwamAIModel(model_config)
    
    dataset = load_dataset(config.data.dataset_name, "main")
    train_texts = [f"{example['question']}\n{example['answer']}" for example in dataset["train"]]
    
    temp_file = "gsm8k_train_temp.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_texts))
    
    math_special_tokens = ["<answer>", "<step>", "<equation>", "<result>", "<reasoning>"]
    
    try:
        logger.info("Training tokenizer with math special tokens")
        tokenizer = VishwamAITokenizer(vocab_size=config.model.vocab_size, special_tokens=math_special_tokens)
        tokenizer.train([temp_file], "tokenizer_output")
    except Exception as e:
        logger.error(f"Tokenizer training failed: {str(e)}")
        try:
            logger.info("Trying minimal tokenizer configuration")
            tokenizer = VishwamAITokenizer(vocab_size=config.model.vocab_size)
            tokenizer.train([temp_file], "tokenizer_output_minimal")
        except Exception as e2:
            logger.error(f"Minimal tokenizer training failed: {str(e2)}")
            raise RuntimeError(f"Could not train tokenizer: {str(e2)}")
    
    os.remove(temp_file)
    logger.info("Tokenizer training completed successfully")
    
    tot = TreeOfThoughts(
        transformer=model,
        tokenizer=tokenizer,
        max_thoughts=config.training.get('tot_max_thoughts', 5),
        max_depth=config.training.get('tot_max_depth', 3),
        beam_width=config.training.get('tot_beam_width', 5)
    )
    model.tot_model = tot
    
    error_trainer = ErrorCorrectionTrainer(
        config=config,
        transformer=model,
        tokenizer=tokenizer,
        use_tot=config.training.get('use_tot', True),
        history_size=config.training.get('error_history_size', 100),
        threshold_percentile=config.training.get('error_threshold_percentile', 85.0)
    )
    
    mesh, sharding = setup_tpu_cluster()
    
    train_dataloader = create_gsm8k_dataloader(config, tokenizer, "train")
    val_dataloader = create_gsm8k_dataloader(config, tokenizer, "test")
    
    checkpoint_dir = config.checkpointing.dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    processor = GSM8KProcessor(tokenizer, config)
    metrics_tracker = {
        'train_loss': [], 'val_loss': [], 'step_accuracy': [], 'exact_match': [], 'answer_match': [], 'tot_score': []
    }
    
    with mesh:
        rng = jax.random.PRNGKey(config.training.seed)
        dummy_input = jnp.ones((1, config.data.max_seq_length), dtype=jnp.int32)
        state = create_train_state(model, config, rng)
        
        dummy_input = jnp.ones((1, config.data.max_seq_length, config.model.hidden_size))
        error_trainer.init_params(rng, dummy_input)
        
        final_state = train(
            model,
            config,
            tokenizer,
            train_dataloader,
            val_dataloader=val_dataloader,
            num_steps=config.training.max_steps,
            log_every=config.monitoring.log_every_n_steps,
            eval_every=config.evaluation.eval_steps,
            checkpoint_dir=checkpoint_dir,
            accum_steps=config.training.get('accum_steps', 1)
        )
        
        val_metrics = {'loss': 0.0, 'step_accuracy': 0.0, 'exact_match': 0.0, 'answer_match': 0.0, 'tot_score': 0.0}
        val_steps = 0
        for batch in val_dataloader():
            eval_outputs = final_state.apply_fn(
                {'params': final_state.params},
                batch['input_ids'],
                attention_mask=batch['attention_mask'],
                deterministic=True,
                use_tot=True
            )
            
            correction_outputs = error_trainer.apply_error_correction(
                logits=eval_outputs['logits'],
                features=eval_outputs['hidden_states'],
                labels=batch.get('labels'),
                training=False,
                rng_key=rng
            )
            
            preds = tokenizer.decode(correction_outputs['corrected_logits'].argmax(-1).tolist())
            targets = tokenizer.decode(batch['labels'].tolist())
            step_metrics = processor.evaluate_step_accuracy(preds, targets)
            
            for k, v in step_metrics.items():
                val_metrics[k] += v
            if 'tot_outputs' in eval_outputs and eval_outputs['tot_outputs'].get('thought'):
                val_metrics['tot_score'] += eval_outputs['tot_outputs']['thought'].score
            val_steps += 1
        
        for k in val_metrics:
            val_metrics[k] /= val_steps
        
        final_metrics = {
            'val_loss': final_state.best_metrics['loss'],
            'step_accuracy': val_metrics['step_accuracy'],
            'exact_match': val_metrics['exact_match'],
            'answer_match': val_metrics['answer_match'],
            'tot_score': val_metrics['tot_score'],
            'train_loss': final_state.best_metrics.get('loss', float('inf')),
            'total_steps': final_state.step
        }
        
        save_model_safetensors(
            final_state.params,
            final_metrics,
            os.path.join(checkpoint_dir, "gsm8k_final.safetensors")
        )
    
    logger.info("Training completed!")
    for k, v in final_metrics.items():
        logger.info(f"Final {k}: {v:.4f}" if isinstance(v, float) else f"Final {k}: {v}")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception("Training failed")
        raise
