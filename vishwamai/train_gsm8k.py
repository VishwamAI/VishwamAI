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
from huggingface_hub import snapshot_download
import safetensors.flax as stf
from omegaconf import OmegaConf, DictConfig
import json
logger = logging.getLogger(__name__)

from vishwamai.training import train, create_train_state
from vishwamai.tokenizer import VishwamAITokenizer
from vishwamai.error_correction import ErrorCorrectionTrainer
from vishwamai.tot import TreeOfThoughts, Thought
from vishwamai.model import (
    ModelArgs, 
    ModelConfig, 
    TransformerBlock, 
    ParallelDense,
    RMSNorm, 
    MoELayer,
    MultiheadAttention,
    ParallelMLP
)

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
    use_gqa: bool = True

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

class VishwamAIModel(nn.Module):
    """Main transformer model with ToT integration."""
    config: ModelConfig

    def setup(self):
        self.embeddings = nn.Embed(
            num_embeddings=self.config.vocab_size,
            features=self.config.hidden_size,
            embedding_init=nn.initializers.normal(stddev=0.02),
            dtype=jnp.dtype(self.config.dtype)
        )
        n_kv_heads = self.config.num_key_value_heads if self.config.use_gqa else self.config.num_attention_heads
        self.encoder = [TransformerBlock(ModelArgs(
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
            use_rope=self.config.use_rope,
            use_flash_attention=self.config.use_flash_attention,
            use_alibi=self.config.use_alibi,
            use_gqa=self.config.use_gqa
        )) for _ in range(self.config.num_layers)]
        self.final_layer_norm = RMSNorm(epsilon=self.config.layer_norm_eps, dtype=jnp.dtype(self.config.dtype))
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
        """Load model weights with optimized memory management."""
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
                        params[name] = jax.device_put(jnp.array(tensor, dtype=dtype))
        self.bind({'params': params})
        return self

class GSM8KProcessor:
    """Processor for GSM8K dataset with robust tokenization and evaluation."""
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
        
        pred_answer = prediction.split('####')[-1].strip() if '####' in prediction else (pred_steps[-1].split()[-1] if pred_steps else "")
        target_answer = target.split('####')[-1].strip() if '####' in target else (target_steps[-1].split()[-1] if target_steps else "")
        
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
            
            if not isinstance(questions, list) or not isinstance(answers, list):
                raise ValueError("Questions and answers must be lists")
            
            if len(questions) != len(answers):
                raise ValueError(f"Mismatched lengths: questions={len(questions)}, answers={len(answers)}")
            
            formatted_texts = []
            for q, a in zip(questions, answers):
                if not isinstance(q, str) or not isinstance(a, str):
                    continue
                formatted_text = f"Question: {q}\nLet's solve this step by step:\n"
                solution_parts = a.split('####')
                steps = solution_parts[0].strip().split('\n') if solution_parts else []
                final_answer = solution_parts[1].strip() if len(solution_parts) > 1 else (steps[-1].strip() if steps else "")
                
                formatted_steps = [f"Step: {step.strip()}" for step in steps if step.strip()]
                formatted_text += f"{chr(10).join(formatted_steps)}\nTherefore, the final answer is: {final_answer}\n####\n{final_answer}"
                formatted_texts.append(formatted_text)
            
            if not formatted_texts:
                raise ValueError("No valid question-answer pairs to tokenize")
            
            tokenized = self.tokenizer(
                formatted_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
            )
            tokenized["labels"] = tokenized["input_ids"].copy()
            tokenized["error_weights"] = np.ones_like(tokenized["input_ids"], dtype=np.float32)
            if formatted_texts:
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

def save_model_safetensors(params: Dict, metrics: Dict, save_path: str, config: Optional[Dict] = None):
    """Save model parameters with metrics and optional config."""
    numpy_params = jax.tree_map(np.asarray, params)
    metadata = {f"metric_{k}": str(v) for k, v in metrics.items()}
    metadata["timestamp"] = str(np.datetime64('now'))
    if config:
        metadata["config"] = json.dumps(config)
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_file(numpy_params, save_path, metadata=metadata)
    logger.info(f"Saved model to {save_path} with metrics: {metrics}")

def setup_tpu_cluster():
    devices = jax.devices()
    logger.info(f"Available devices: {devices}")
    device_count = len(devices)
    if device_count < 2:
        logger.warning("Limited devices detected; parallelism may be reduced")
    device_mesh = np.array(devices).reshape((device_count // 2, 2)) if device_count >= 2 else np.array(devices).reshape((1, device_count))
    mesh = Mesh(device_mesh, ('data', 'model'))
    sharding = NamedSharding(mesh, P('data', 'model'))
    return mesh, sharding

def validate_config(config: DictConfig):
    """Validate configuration schema."""
    schema = OmegaConf.create({
        'model': {
            'vocab_size': int,
            'hidden_size': int,
            'num_layers': int,
            'num_attention_heads': int,
            'intermediate_size': int,
            'hidden_dropout_prob': float,
            'attention_dropout_prob': float,
            'max_position_embeddings': int,
            'use_gqa': bool,
            'num_key_value_heads': int,
            'dtype': str
        },
        'training': {
            'max_steps': int,
            'seed': int,
            'eval_every': int,
            'log_every_n_steps': int,
            'use_tot': bool,
            'tot_max_thoughts': int,
            'tot_max_depth': int,
            'tot_beam_width': int
        },
        'data': {
            'batch_size': int,
            'max_seq_length': int,
            'dataset_name': str
        },
        'checkpointing': {
            'dir': str
        },
        'monitoring': {
            'log_every_n_steps': int
        },
        'evaluation': {
            'eval_steps': int
        }
    })
    try:
        OmegaConf.merge(schema, config)
    except Exception as e:
        logger.error(f"Config validation failed: {str(e)}")
        raise

def main(config_path: str = "vishwamai/configs/training/gsm8k.yaml"):
    """Main training function with enhanced error handling and validation."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    try:
        config = OmegaConf.load(config_path)
        validate_config(config)
    except FileNotFoundError:
        logger.error(f"Config file not found at {config_path}")
        raise
    except ValueError as e:
        logger.error(f"Invalid configuration: {str(e)}")
        raise
    
    model_params = {k: v for k, v in config.model.items() if k in ModelConfig.__dataclass_fields__}
    try:
        model_config = ModelConfig(**model_params)
    except ValueError as e:
        logger.error(f"Invalid model configuration: {str(e)}")
        raise
    
    model = VishwamAIModel(model_config)
    
    dataset = load_dataset(config.data.dataset_name, "main")
    train_texts = [f"{example['question']}\n{example['answer']}" for example in dataset["train"]]
    
    temp_file = "gsm8k_train_temp.txt"
    try:
        with open(temp_file, "w", encoding="utf-8") as f:
            f.write("\n".join(train_texts))
    except IOError as e:
        logger.error(f"Failed to write temporary file: {str(e)}")
        raise
    
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
            val_metrics[k] /= val_steps if val_steps > 0 else 1
        
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
            os.path.join(checkpoint_dir, "gsm8k_final.safetensors"),
            config=OmegaConf.to_container(config)
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