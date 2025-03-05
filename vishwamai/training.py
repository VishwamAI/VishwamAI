"""
Enhanced training module for VishwamAI models.
"""

import jax
import jax.numpy as jnp
import flax
import optax
from typing import Dict, List, Tuple, Callable, Any, Iterator, Optional
import numpy as np
from datasets import load_dataset
from omegaconf import OmegaConf, DictConfig
import random
import logging
from functools import partial
import os
from .tot import TreeOfThoughts
from .model import VishwamAIModel, ModelConfig
from .loss_functions import cross_entropy_loss, tot_guided_loss, compute_metrics, compute_composite_loss
from .error_correction import ErrorCorrectionTrainer, create_error_corrected_train_step, create_error_corrected_eval_step
from .tokenizer import VishwamAITokenizer

logger = logging.getLogger(__name__)

class VishwamaiTrainingState:
    """Enhanced training state with additional features."""
    apply_fn: Callable
    params: Dict
    tx: optax.GradientTransformation
    opt_state: Any
    ema_params: Optional[Dict]
    step: int
    best_metrics: Dict
    tot_state: Dict
    error_state: Dict
    
    @classmethod
    def create(cls, apply_fn, params, tx, ema_params=None, best_metrics=None, tot_state=None, error_state=None):
        return cls(
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=tx.init(params),
            ema_params=ema_params,
            step=0,
            best_metrics=best_metrics if best_metrics is not None else {},
            tot_state=tot_state if tot_state is not None else {},
            error_state=error_state if error_state is not None else {}
        )
    
    def __init__(self, apply_fn, params, tx, opt_state, ema_params=None, step=0, best_metrics=None, tot_state=None, error_state=None):
        self.apply_fn = apply_fn
        self.params = params
        self.tx = tx
        self.opt_state = opt_state
        self.ema_params = ema_params
        self.step = step
        self.best_metrics = best_metrics if best_metrics is not None else {}
        self.tot_state = tot_state if tot_state is not None else {}
        self.error_state = error_state if error_state is not None else {}
    
    def apply_gradients(self, grads, **kwargs):
        updates, new_opt_state = self.tx.update(grads, self.opt_state, self.params)
        new_params = optax.apply_updates(self.params, updates)
        return self.replace(
            params=new_params,
            opt_state=new_opt_state,
            step=self.step + 1,
            **kwargs
        )
    
    def replace(self, **kwargs):
        return self.__class__(
            apply_fn=kwargs.get('apply_fn', self.apply_fn),
            params=kwargs.get('params', self.params),
            tx=kwargs.get('tx', self.tx),
            opt_state=kwargs.get('opt_state', self.opt_state),
            ema_params=kwargs.get('ema_params', self.ema_params),
            step=kwargs.get('step', self.step),
            best_metrics=kwargs.get('best_metrics', self.best_metrics),
            tot_state=kwargs.get('tot_state', self.tot_state),
            error_state=kwargs.get('error_state', self.error_state)
        )

TrainingState = VishwamaiTrainingState

def create_learning_rate_scheduler(
    factors="constant * linear_warmup * cosine_decay",
    base_learning_rate=0.0001,
    warmup_steps=1000,
    decay_steps=100000,
):
    factors = [f.strip() for f in factors.split('*')]
    
    def schedule(step):
        rate = 1.0
        for factor in factors:
            if factor == 'constant':
                rate *= base_learning_rate
            elif factor == 'linear_warmup':
                rate *= jnp.minimum(1.0, step / warmup_steps)
            elif factor == 'cosine_decay':
                rate *= 0.5 * (1 + jnp.cos(jnp.pi * jnp.minimum(step, decay_steps) / decay_steps))
        return rate
    
    return schedule

def create_optimizer(config):
    """Create optimizer with mixed precision and TPU optimization support."""
    lr_schedule = create_learning_rate_scheduler(
        base_learning_rate=config.training.learning_rate,
        warmup_steps=config.training.warmup_steps,
        decay_steps=config.training.max_steps,
    )
    
    def loss_scale_fn(step):
        """Dynamic loss scaling for mixed precision training."""
        return jnp.where(
            step < config.training.warmup_steps,
            1.0,  # Use normal scaling during warmup
            128.0  # Use higher scaling after warmup
        )
    
    tx = optax.chain(
        optax.clip_by_global_norm(config.training.max_grad_norm),
        # Enable mixed precision training
        optax.scale_by_schedule(loss_scale_fn),
        # Use Adam optimizer with parameters tuned for TPU
        optax.scale_by_adam(
            b1=config.training.adam_beta1,
            b2=config.training.adam_beta2,
            eps=1e-6,  # Increased epsilon for TPU stability
            eps_root=1e-6
        ),
        # Apply weight decay after Adam
        optax.add_decayed_weights(config.training.weight_decay),
        # Scale learning rate
        optax.scale_by_schedule(lr_schedule),
        # Final scaling
        optax.scale(-1.0)
    )
    
    return tx, lr_schedule

def create_train_state(model, config, rng: jax.random.PRNGKey, mesh: Optional[Any] = None) -> TrainingState:
    """Create training state with optional device mesh support."""
    tx, _ = create_optimizer(config)

    # Default device setup for TPU/GPU
    if mesh is None:
        # Try to get default TPU/GPU device
        try:
            devices = jax.local_devices()
            if devices:
                mesh = devices[0]
        except:
            pass
    
    if not hasattr(model, 'params') or model.params is None:
        dummy_input = jnp.ones((1, config.data.max_seq_length), dtype=jnp.int32)
        dummy_attention_mask = jnp.ones((1, config.data.max_seq_length), dtype=jnp.int32)
        # Handle both mesh and single device cases
        device = None
        if mesh is not None:
            if hasattr(mesh, 'devices_flat'):
                device = mesh.devices_flat[0]  # Mesh object case
            else:
                device = mesh  # Single device case
        
        with jax.default_device(device):
            params = model.init(rng, dummy_input, attention_mask=dummy_attention_mask)['params']
    else:
        params = model.params
    
    state = TrainingState.create(
        apply_fn=model.__call__,
        params=params,
        tx=tx,
        ema_params=params if config.training.use_ema else None,
        best_metrics={
            'loss': float('inf'),
            'accuracy': 0.0,
            'ec_improvement': 0.0,
            'tot_score': 0.0
        },
        tot_state={
            'enabled': config.training.get('use_tot', False),
            'search_strategy': config.training.get('tot_search_strategy', 'beam'),
            'thoughts_per_batch': 0,
            'best_thought_score': 0.0
        },
        error_state={
            'enabled': config.training.get('use_error_correction', True),
            'history_size': config.training.get('error_history_size', 100),
            'threshold': 0.1
        }
    )
    
    return state

class DataProcessor:
    """Enhanced processor with curriculum learning support."""
    def __init__(self, tokenizer: VishwamAITokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.data.max_seq_length
        self.curriculum_scheduler = self._create_curriculum_scheduler()
        
    def _create_curriculum_scheduler(self):
        """Create curriculum learning scheduler."""
        if not hasattr(self.config.training, 'curriculum'):
            return None
        
        curr_config = self.config.training.curriculum
        curr_type = curr_config.get('type', 'length')
        
        if curr_type == 'length':
            return {
                'current_max_length': curr_config.get('initial_length', 64),
                'max_length': self.max_length,
                'length_increment': curr_config.get('length_increment', 32),
                'update_every': curr_config.get('update_every', 1000),
                'steps': 0
            }
        return None
    
    def update_curriculum(self):
        """Update curriculum difficulty."""
        if not self.curriculum_scheduler:
            return
        
        self.curriculum_scheduler['steps'] += 1
        if self.curriculum_scheduler['steps'] % self.curriculum_scheduler['update_every'] == 0:
            current_length = self.curriculum_scheduler['current_max_length']
            max_length = self.curriculum_scheduler['max_length']
            increment = self.curriculum_scheduler['length_increment']
            
            new_length = min(current_length + increment, max_length)
            self.curriculum_scheduler['current_max_length'] = new_length
            logger.info(f"Curriculum updated: sequence length = {new_length}")
    
    def tokenize_function(self, examples):
        questions = examples['question']
        answers = examples['answer']
        formatted_texts = [f"Question: {q}\nAnswer: {a}" for q, a in zip(questions, answers)]
        
        # Get current max length from curriculum if enabled
        max_length = (self.curriculum_scheduler['current_max_length'] 
                     if self.curriculum_scheduler 
                     else self.max_length)
        
        tokenized = self.tokenizer(
            formatted_texts,
            padding="max_length",
            truncation=True,
            max_length=max_length,
            return_attention_mask=True,
            return_tensors="np",
        )
        
        # Enable mixed precision training by using float16
        tokenized["labels"] = tokenized["input_ids"].copy()
        tokenized["attention_mask"] = tokenized["attention_mask"].astype(np.float16)
        tokenized["error_weights"] = np.ones_like(tokenized["input_ids"], dtype=np.float16)
        
        return tokenized
    
    def prepare_dataset(self, dataset):
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            num_proc=self.config.data.preprocessing_num_workers,
            remove_columns=dataset.column_names,
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

def create_train_dataloader(config, tokenizer):
    dataset = load_dataset("openai/gsm8k", "main", split=config.data.train_split)
    data_processor = DataProcessor(tokenizer, config)
    processed_dataset = data_processor.prepare_dataset(dataset)
    
    def data_iterator():
        epoch = 0
        while True:
            indices = list(range(len(processed_dataset)))
            random.shuffle(indices)
            for i in range(0, len(indices), config.training.batch_size):
                batch_indices = indices[i:i + config.training.batch_size]
                examples = [processed_dataset[idx] for idx in batch_indices]
                yield data_processor.collate_fn(examples)
            epoch += 1
            logger.info(f"Finished epoch {epoch}")
    
    return data_iterator()

def create_val_dataloader(config, tokenizer):
    dataset = load_dataset("openai/gsm8k", "main", split=config.data.val_split)
    data_processor = DataProcessor(tokenizer, config)
    processed_dataset = data_processor.prepare_dataset(dataset)
    
    def data_iterator():
        indices = list(range(len(processed_dataset)))
        for i in range(0, len(indices), config.training.eval_batch_size):
            batch_indices = indices[i:i + config.training.eval_batch_size]
            examples = [processed_dataset[idx] for idx in batch_indices]
            yield data_processor.collate_fn(examples)
    
    return data_iterator()

@partial(jax.jit, static_argnums=(3, 4))
def train_step(
    state: TrainingState,
    batch: Dict[str, jnp.ndarray],
    model_config: ModelConfig,
    z_loss: float = 0.0,
    rng_key: Optional[jax.random.PRNGKey] = None,
    accum_steps: int = 1
) -> Tuple[TrainingState, Dict]:
    """JAX-jitted training step with mixed precision support."""
    use_tot = state.tot_state.get('enabled', False)
    use_mod = model_config.use_mod
    
    def loss_fn(params):
        step_key, dropout_key, tot_key = jax.random.split(rng_key, 3) if rng_key else (None, None, None)
        
        # Ensure inputs are in optimal format for TPU
        input_ids = batch['input_ids']
        attention_mask = batch['attention_mask'].astype(jnp.float16)  # Mixed precision
        
        outputs = state.apply_fn(
            {'params': params},
            input_ids,
            attention_mask=attention_mask,
            deterministic=False,
            rngs={'dropout': dropout_key},
            use_tot=use_tot,
            tot_rng_key=tot_key if use_tot else None,
            # Enable mixed precision in model forward pass
            precision=jax.lax.Precision.DEFAULT
        )
        
        logits = outputs.get('logits')
        if logits is None:
            raise ValueError("Model output doesn't contain 'logits'")
        
        # Cast logits to float32 for loss computation stability
        shift_logits = logits[:, :-1, :].astype(jnp.float32)
        shift_labels = batch['labels'][:, 1:]
        
        loss, metrics = compute_composite_loss(
            outputs=outputs,
            batch={'labels': shift_labels},
            weights={
                'ce': model_config.get('ce_weight', 0.5),
                'kd': model_config.get('kd_weight', 0.0),
                'tot': model_config.get('tot_weight', 0.3) if use_tot else 0.0,
                'moe': model_config.get('moe_weight', 0.2) if use_mod else 0.0
            }
        )
        
        if z_loss > 0:
            z_loss_term = z_loss * jnp.mean(jnp.sum(shift_logits ** 2, axis=-1))
            loss += z_loss_term
            metrics['z_loss'] = float(z_loss_term)
        
        return loss, metrics
    
    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    accum_grads = jax.tree_map(jnp.zeros_like, state.params)
    for _ in range(accum_steps):
        step_key, sub_key = jax.random.split(rng_key) if rng_key else (None, None)
        (loss, metrics), grads = grad_fn(state.params)
        accum_grads = jax.tree_map(lambda a, g: a + g / accum_steps, accum_grads, grads)
        rng_key = sub_key
    
    new_state = state.apply_gradients(grads=accum_grads)
    
    if new_state.ema_params is not None:
        new_ema_params = jax.tree_map(
            lambda ema, param: ema * 0.999 + param * 0.001,
            new_state.ema_params, new_state.params
        )
        new_state = new_state.replace(ema_params=new_ema_params)
    
    # Update ToT state if enabled
    if use_tot and 'tot_outputs' in metrics:
        new_state = new_state.replace(
            tot_state={
                **new_state.tot_state,
                'thoughts_per_batch': metrics['tot_outputs'].get('thoughts_per_batch', 0),
                'best_thought_score': metrics['tot_outputs'].get('score', 0.0)
            }
        )
    
    return new_state, metrics

def eval_step(
    state: TrainingState,
    batch: Dict[str, jnp.ndarray],
    use_tot: bool = False,
    rng_key: Optional[jax.random.PRNGKey] = None
) -> Dict:
    dropout_key, tot_key = jax.random.split(rng_key, 2) if rng_key else (None, None)
    
    outputs = state.apply_fn(
        {'params': state.params},
        batch['input_ids'],
        attention_mask=batch['attention_mask'],
        deterministic=True,
        rngs={'dropout': dropout_key} if dropout_key else None,
        use_tot=use_tot,
        tot_rng_key=tot_key if use_tot else None
    )
    
    logits = outputs.get('logits')
    if logits is None:
        raise ValueError("Model output doesn't contain 'logits'")
    
    shift_logits = logits[:, :-1, :]
    shift_labels = batch['labels'][:, 1:]
    
    metrics = compute_metrics(shift_logits, shift_labels, outputs.get('corrected_logits'))
    
    if use_tot and 'tot_outputs' in outputs:
        tot_outputs = outputs['tot_outputs']
        metrics.update({
            'tot_score': float(tot_outputs.get('thought', {}).score if 'thought' in tot_outputs else 0.0),
            'tot_weight': float(tot_outputs.get('integration_info', [0])[0] if 'integration_info' in tot_outputs else 0)
        })
    
    return metrics

def train(
    model: VishwamAIModel,
    config: DictConfig,
    tokenizer: VishwamAITokenizer,
    train_dataloader: Iterator,
    val_dataloader: Optional[Iterator] = None,
    num_steps: int = 10000,
    log_every: int = 100,
    eval_every: int = 1000,
    checkpoint_dir: Optional[str] = None,
    accum_steps: int = 1,
    mesh: Optional[Any] = None
) -> TrainingState:
    logger.info("Starting enhanced training with ToT and error correction")
    
    use_error_correction = config.training.get('use_error_correction', True)
    if use_error_correction:
        error_trainer = ErrorCorrectionTrainer(
            config=config,
            transformer=model,
            tokenizer=tokenizer,
            use_tot=config.training.get('use_tot', True),
            use_mod=config.model.get('use_mod', True)
        )
        ec_train_step = create_error_corrected_train_step(train_step, error_trainer)
        ec_eval_step = create_error_corrected_eval_step(eval_step, error_trainer)
    else:
        ec_train_step = train_step
        ec_eval_step = eval_step
        error_trainer = None
    
    rng_key = jax.random.PRNGKey(config.training.seed)
    state = create_train_state(model, config, rng_key, mesh)
    
    if use_error_correction:
        rng_key, init_key = jax.random.split(rng_key)
        # Use smaller dimensions for error correction initialization
        reduced_seq_len = min(config.data.max_seq_length, 128)
        reduced_hidden_size = min(config.model.hidden_size, 512)
        dummy_input = jnp.ones((1, reduced_seq_len, reduced_hidden_size))
        dummy_labels = jnp.ones((1, reduced_seq_len), dtype=jnp.int32)
        error_trainer.init_params(init_key, dummy_input, dummy_labels)
    
    for step in range(num_steps):
        batch = next(train_dataloader)
        rng_key, step_key = jax.random.split(rng_key)
        
        state, metrics = ec_train_step(
            state=state,
            batch=batch,
            model_config=config.model,
            z_loss=config.training.get('z_loss', 0.0),
            rng_key=step_key,
            accum_steps=accum_steps
        )
        
        if step % log_every == 0:
            logger.info(f"Step {step}/{num_steps}: loss = {metrics['loss']:.4f}, accuracy = {metrics['metrics']['accuracy']:.4f}")
            if state.tot_state.get('enabled', False):
                logger.info(f"ToT: thoughts/batch = {state.tot_state['thoughts_per_batch']}, best_score = {state.tot_state['best_thought_score']:.4f}")
            if use_error_correction and 'improvement' in metrics['metrics']:
                logger.info(f"Error Correction: improvement = {metrics['metrics']['improvement']:.4f}")
        
        if val_dataloader and step % eval_every == 0:
            eval_metrics = {}
            for eval_batch in val_dataloader():
                batch_metrics = ec_eval_step(state, eval_batch, use_tot=state.tot_state.get('enabled', False), rng_key=rng_key)
                for k, v in batch_metrics['metrics'].items():
                    eval_metrics[k] = eval_metrics.get(k, 0.0) + v / (len(list(val_dataloader())) + 1e-8)
            
            logger.info(f"Eval at step {step}: { {k: f'{v:.4f}' for k, v in eval_metrics.items()} }")
            
            if eval_metrics['loss'] < state.best_metrics['loss']:
                state = state.replace(best_metrics={
                    'loss': eval_metrics['loss'],
                    'accuracy': eval_metrics['accuracy'],
                    'ec_improvement': eval_metrics.get('improvement', 0.0),
                    'tot_score': eval_metrics.get('tot_score', 0.0)
                })
                if checkpoint_dir:
                    save_checkpoint(state, os.path.join(checkpoint_dir, "best_model"))
        
        if checkpoint_dir and step % config.training.save_every == 0:
            save_checkpoint(state, os.path.join(checkpoint_dir, f"checkpoint_{step}"))
    
    if checkpoint_dir:
        save_checkpoint(state, os.path.join(checkpoint_dir, "final_model"))
    
    return state

def save_checkpoint(state: TrainingState, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(f"{path}.msgpack", "wb") as f:
        f.write(flax.serialization.msgpack_serialize({
            'params': jax.tree_map(np.asarray, state.params),
            'opt_state': state.opt_state,
            'step': state.step,
            'best_metrics': state.best_metrics,
            'tot_state': state.tot_state,
            'error_state': state.error_state
        }))
    logger.info(f"Saved checkpoint to {path}.msgpack")

def main(config_path: str = None, mesh: Optional[Any] = None) -> TrainingState:
    if config_path is None:
        config_path = "vishwamai/configs/training/gsm8k.yaml"
    config = OmegaConf.load(config_path)
    model = VishwamAIModel(ModelConfig(**config.model))
    
    if 'use_error_correction' not in config.training:
        config.training.use_error_correction = True
    
    tokenizer = VishwamAITokenizer(vocab_size=config.model.vocab_size)
    dataset = load_dataset(config.data.dataset_name, "main")
    train_texts = [f"{example['question']}\n{example['answer']}" for example in dataset["train"]]
    
    temp_file = "gsm8k_train_temp.txt"
    with open(temp_file, "w", encoding="utf-8") as f:
        f.write("\n".join(train_texts))
    
    try:
        tokenizer.train([temp_file], "tokenizer_output")
    finally:
        if os.path.exists(temp_file):
            os.remove(temp_file)
    
    train_dataloader = create_train_dataloader(config, tokenizer)
    val_dataloader = create_val_dataloader(config, tokenizer)
    
    checkpoint_dir = config.training.checkpoint_dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    final_state = train(
        model,
        config,
        tokenizer,
        train_dataloader,
        val_dataloader,
        num_steps=config.training.max_steps,
        log_every=config.training.log_every,
        eval_every=config.training.eval_every,
        checkpoint_dir=checkpoint_dir,
        accum_steps=config.training.get('accum_steps', 1),
        mesh=mesh
    )
    
    logger.info(f"Training completed! Best metrics: {final_state.best_metrics}")
    return final_state

if __name__ == "__main__":
    try:
        # Initialize TPU/GPU device mesh if available
        devices = None
        try:
            from jax.experimental import mesh_utils
            devices = mesh_utils.create_device_mesh((8,))  # Default TPU setup
        except:
            try:
                devices = jax.local_devices()  # Fallback to available devices
                if devices:
                    devices = devices[0]
            except:
                pass
        
        # Run main with device configuration
        main(mesh=devices)
    except Exception as e:
        logger.exception("Training failed")
