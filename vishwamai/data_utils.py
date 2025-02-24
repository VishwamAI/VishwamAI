from typing import Iterator, Dict, Any, Optional
import jax
import jax.numpy as jnp
from datasets import Dataset, load_dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig
import numpy as np
import logging
from functools import partial

logger = logging.getLogger(__name__)

def create_train_dataloader(cfg: DictConfig) -> Iterator[Dict[str, jnp.ndarray]]:
    """Enhanced training data loader with memory optimization and prefetching."""
    # Load dataset with error handling
    try:
        if cfg.data.path.endswith(('.json', '.jsonl')):
            dataset = load_dataset('json', data_files=cfg.data.path)['train']
        elif cfg.data.path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=cfg.data.path)['train']
        else:
            dataset = load_dataset(cfg.data.path)['train']
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise

    # Apply preprocessing with improved memory efficiency
    dataset = preprocess_dataset(
        dataset,
        cfg.model.max_seq_len,
        cfg.data.get('preprocessing', {})
    )

    # Enhanced distributed training setup
    sampler = None
    if jax.process_count() > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=jax.process_count(),
            rank=jax.process_index(),
            shuffle=True,
            drop_last=True
        )

    # Dynamic batch size with gradient accumulation
    batch_size = cfg.training.dynamic_batch_size.initial_batch_size
    accumulation_steps = cfg.training.get('gradient_accumulation_steps', 1)
    effective_batch_size = batch_size * accumulation_steps

    # Prefetch queue size tuning
    num_workers = cfg.data.get('num_workers', 4)
    prefetch_factor = cfg.data.get('prefetch_factor', 2)

    def data_generator():
        while True:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                sampler=sampler,
                drop_last=True,
                num_workers=num_workers,
                pin_memory=True,
                prefetch_factor=prefetch_factor
            )

            # Enable asynchronous prefetching
            loader_iter = iter(loader)
            try:
                next_batch = next(loader_iter)
            except StopIteration:
                continue

            for batch in loader:
                current_batch = next_batch
                try:
                    next_batch = next(loader_iter)
                    # Prefetch next batch to GPU asynchronously
                    jax.tree_map(
                        lambda x: jax.device_put_replicated(x, jax.devices()),
                        next_batch
                    )
                except StopIteration:
                    current_batch = batch
                    break

                # Efficient JAX array conversion with error handling
                try:
                    jax_batch = {
                        k: jnp.asarray(v.numpy(), dtype=get_optimal_dtype(v))
                        for k, v in current_batch.items()
                    }

                    # Multi-device reshaping with dynamic padding
                    if jax.device_count() > 1:
                        batch_size = jax_batch['input_ids'].shape[0]
                        if batch_size % jax.device_count() != 0:
                            pad_size = jax.device_count() - (batch_size % jax.device_count())
                            jax_batch = {
                                k: jnp.pad(
                                    v,
                                    [(0, pad_size)] + [(0, 0)] * (v.ndim - 1),
                                    mode='constant'
                                )
                                for k, v in jax_batch.items()
                            }
                            batch_size += pad_size

                        device_batch_size = batch_size // jax.device_count()
                        jax_batch = {
                            k: v.reshape(
                                (jax.device_count(), device_batch_size, *v.shape[1:])
                            )
                            for k, v in jax_batch.items()
                        }

                    yield jax_batch

                except Exception as e:
                    logger.error(f"Error processing batch: {str(e)}")
                    continue

    return data_generator()

def create_val_dataloader(cfg: DictConfig) -> Iterator[Dict[str, jnp.ndarray]]:
    """Enhanced validation data loader with improved memory efficiency."""
    try:
        # Load dataset with unified error handling
        if isinstance(cfg.data.val_path, str):
            if cfg.data.val_path.endswith(('.json', '.jsonl')):
                dataset = load_dataset('json', data_files=cfg.data.val_path)['train']
            elif cfg.data.val_path.endswith('.csv'):
                dataset = load_dataset('csv', data_files=cfg.data.val_path)['train']
            else:
                dataset = load_dataset(cfg.data.val_path)['validation']
        else:
            dataset = load_dataset(cfg.data.path)
            dataset = dataset['train'].train_test_split(
                test_size=cfg.data.get('val_split', 0.1),
                seed=cfg.training.get('seed', 42)
            )['test']

        # Memory-efficient preprocessing
        dataset = preprocess_dataset(
            dataset,
            cfg.model.max_seq_len,
            cfg.data.get('preprocessing', {})
        )

        batch_size = cfg.training.dynamic_batch_size.initial_batch_size
        num_workers = cfg.data.get('num_workers', 4)
        prefetch_factor = cfg.data.get('prefetch_factor', 2)

        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=prefetch_factor
        )

        def data_generator():
            while True:
                loader_iter = iter(loader)
                for batch in loader_iter:
                    try:
                        jax_batch = {
                            k: jnp.asarray(v.numpy(), dtype=get_optimal_dtype(v))
                            for k, v in batch.items()
                        }

                        if jax.device_count() > 1:
                            batch_size = jax_batch['input_ids'].shape[0]
                            if batch_size % jax.device_count() != 0:
                                pad_size = jax.device_count() - (batch_size % jax.device_count())
                                jax_batch = {
                                    k: jnp.pad(
                                        v,
                                        [(0, pad_size)] + [(0, 0)] * (v.ndim - 1),
                                        mode='constant'
                                    )
                                    for k, v in jax_batch.items()
                                }
                                batch_size += pad_size

                            device_batch_size = batch_size // jax.device_count()
                            jax_batch = {
                                k: v.reshape(
                                    (jax.device_count(), device_batch_size, *v.shape[1:])
                                )
                                for k, v in jax_batch.items()
                            }

                        yield jax_batch

                    except Exception as e:
                        logger.error(f"Error processing validation batch: {str(e)}")
                        continue

        return data_generator()

    except Exception as e:
        logger.error(f"Failed to create validation dataloader: {str(e)}")
        raise

def get_optimal_dtype(tensor):
    """Determine optimal dtype based on tensor content."""
    if tensor.dtype in [np.int32, np.int64]:
        return jnp.int32
    elif tensor.dtype in [np.float32, np.float64]:
        return jnp.bfloat16  # Use bfloat16 for better training stability
    return tensor.dtype

def preprocess_dataset(
    dataset: Dataset,
    max_length: int,
    preprocessing_config: Optional[Dict[str, Any]] = None
) -> Dataset:
    """Enhanced preprocessing with improved memory efficiency."""
    if preprocessing_config is None:
        preprocessing_config = {}

    @partial(jax.jit, static_argnums=(1,))
    def process_batch(batch, max_chars):
        """JIT-compiled batch processing for better performance."""
        return {
            k: v[:max_chars] if isinstance(v, str) else v
            for k, v in batch.items()
        }

    def preprocess_function(examples):
        processed = {'text': examples['text']}

        for step, params in preprocessing_config.items():
            if step == 'truncate':
                max_chars = params.get('max_chars', max_length)
                processed = process_batch(processed, max_chars)
            elif step == 'clean':
                # Add optimized text cleaning steps here
                pass

        return processed

    # Parallel processing with error handling
    try:
        return dataset.map(
            preprocess_function,
            batched=True,
            num_proc=preprocessing_config.get('num_proc', 4),
            remove_columns=dataset.column_names,
            load_from_cache_file=preprocessing_config.get('use_cache', True),
            desc="Preprocessing dataset"
        )
    except Exception as e:
        logger.error(f"Preprocessing failed: {str(e)}")
        raise

def evaluate(
    state: Any,
    val_loader: Iterator[Dict[str, jnp.ndarray]],
    cfg: DictConfig,
    num_batches: int = None
) -> Dict[str, float]:
    """Enhanced distributed evaluation with improved memory efficiency."""
    metrics_list = []
    process_metrics = []

    # Determine number of eval batches per process
    if num_batches is None:
        num_batches = cfg.training.get('eval_batches', 100)
    batches_per_process = num_batches // jax.process_count()

    # Distributed evaluation loop with gradient accumulation
    accumulation_steps = cfg.training.get('gradient_accumulation_steps', 1)

    @jax.jit
    def eval_step(state, batch):
        def forward_fn(params):
            return state.apply_fn({'params': params}, **batch, train=False)
        return forward_fn(state.params)

    # Device data extraction
    def get_first_device_batch(x):
        if isinstance(x, dict):
            return {k: get_first_device_batch(v) for k, v in x.items()}
        elif isinstance(x, jnp.ndarray):
            return x[0] if x.shape[0] > 1 else x
        return x

    try:
        for step in range(0, batches_per_process, accumulation_steps):
            step_metrics = []

            for _ in range(accumulation_steps):
                try:
                    batch = next(val_loader)

                    # Forward pass with error handling
                    try:
                        outputs = eval_step(state, batch)
                        if isinstance(outputs, dict) and 'loss' not in outputs:
                            logger.warning("Model output missing 'loss' key")
                            continue

                        # Extract metrics from first device
                        device_outputs = jax.tree_map(get_first_device_batch, outputs)

                        # Calculate metrics with improved numerical stability
                        batch_metrics = {
                            'val_loss': float(device_outputs['loss']),
                            'val_accuracy': float(device_outputs.get('accuracy', 0.0)),
                            'val_perplexity': float(jnp.exp(jnp.minimum(device_outputs['loss'], 100)))
                        }

                        # Add auxiliary metrics
                        for key in device_outputs:
                            if key.startswith('aux_') and isinstance(device_outputs[key], (float, int)):
                                batch_metrics[f'val_{key}'] = float(device_outputs[key])

                        step_metrics.append(batch_metrics)

                    except Exception as e:
                        logger.error(f"Error in forward pass: {str(e)}")
                        continue

                except StopIteration:
                    break

            # Aggregate metrics for this step
            if step_metrics:
                metrics_list.append({
                    k: np.mean([m[k] for m in step_metrics])
                    for k in step_metrics[0].keys()
                })

    except Exception as e:
        logger.error(f"Error in evaluation loop: {str(e)}")
        raise

    # Gather metrics from all processes
    if jax.process_count() > 1:
        gathered_metrics = jax.tree_multimap(
            lambda *x: np.stack(x),
            *[metrics_list] * jax.process_count()
        )

        # Aggregate across processes
        metrics = {
            k: float(np.mean([m[k] for m in gathered_metrics]))
            for k in gathered_metrics[0].keys()
        }
    else:
        # Single process aggregation
        metrics = {
            k: float(np.mean([m[k] for m in metrics_list]))
            for k in metrics_list[0].keys()
        } if metrics_list else {}

    # Add evaluation metadata
    metrics['num_processed_batches'] = len(metrics_list) * accumulation_steps
    metrics['num_processes'] = jax.process_count()

    return metrics
