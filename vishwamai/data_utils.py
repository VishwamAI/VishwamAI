"""Data loading and preprocessing utilities optimized for TPU."""
from typing import Iterator, Dict, Any, Optional, Callable
import os
import jax
import jax.numpy as jnp
from datasets import Dataset, load_dataset, DatasetDict
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from omegaconf import DictConfig
import numpy as np
import logging
from functools import partial
from google.cloud import storage

logger = logging.getLogger(__name__)

def get_gcs_dataset(path: str) -> str:
    """Load dataset from Google Cloud Storage."""
    if not path.startswith("gs://"):
        return path
        
    client = storage.Client()
    bucket_name, blob_path = path.replace("gs://", "").split("/", 1)
    bucket = client.bucket(bucket_name)
    local_path = f"/tmp/{os.path.basename(path)}"
    
    # Download dataset file
    blob = bucket.blob(blob_path)
    blob.download_to_filename(local_path)
    return local_path

@partial(jax.jit, static_argnums=(1,))
def create_data_parallel_batch(batch: Dict[str, jnp.ndarray], num_devices: int) -> Dict[str, jnp.ndarray]:
    """Reshape batch for data-parallel processing on TPU."""
    return jax.tree_map(
        lambda x: x.reshape((num_devices, -1) + x.shape[1:]) if x is not None else None,
        batch
    )

def create_dualpipe_batch(batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """Split batch for dualpipe processing."""
    batch_size = batch['input_ids'].shape[0]
    split_point = batch_size // 2
    
    forward_batch = {k: v[:split_point] for k, v in batch.items()}
    backward_batch = {k: v[split_point:] for k, v in batch.items()}
    
    return forward_batch, backward_batch

def create_train_dataloader(cfg: DictConfig) -> Iterator[Dict[str, jnp.ndarray]]:
    """Enhanced training data loader with TPU optimizations."""
    try:
        # Handle GCS paths
        data_path = get_gcs_dataset(cfg.data.path)
        
        # Load dataset with format detection
        if data_path.endswith(('.json', '.jsonl')):
            dataset = load_dataset('json', data_files=data_path)['train']
        elif data_path.endswith('.csv'):
            dataset = load_dataset('csv', data_files=data_path)['train']
        else:
            dataset = load_dataset(data_path)['train']
            
        # Apply preprocessing with TPU optimization
        dataset = preprocess_dataset(
            dataset,
            cfg.model.max_seq_len,
            cfg.data.get('preprocessing', {})
        )

        # TPU-aware distributed setup
        num_local_devices = jax.local_device_count()
        global_batch_size = cfg.training.dynamic_batch_size.initial_batch_size * num_local_devices
        
        sampler = None
        if jax.process_count() > 1:
            sampler = DistributedSampler(
                dataset,
                num_replicas=jax.process_count(),
                rank=jax.process_index(),
                shuffle=True,
                drop_last=True
            )

        # Optimize worker count for TPU
        num_workers = min(
            cfg.data.get('num_workers', 4),
            os.cpu_count() or 4
        )

        def data_generator():
            while True:
                loader = DataLoader(
                    dataset,
                    batch_size=global_batch_size,
                    sampler=sampler,
                    drop_last=True,
                    num_workers=num_workers,
                    pin_memory=True,
                    prefetch_factor=2
                )

                # TPU-optimized prefetching
                loader_iter = iter(loader)
                try:
                    next_batch = next(loader_iter)
                except StopIteration:
                    continue

                for batch in loader:
                    current_batch = next_batch
                    try:
                        next_batch = next(loader_iter)
                        # Prefetch to TPU
                        jax.tree_map(
                            lambda x: jax.device_put_sharded(
                                list(x.numpy().reshape((num_local_devices, -1) + x.shape[1:])),
                                jax.local_devices()
                            ),
                            next_batch
                        )
                    except StopIteration:
                        current_batch = batch
                        break

                    try:
                        # Convert to JAX arrays with TPU-optimal dtypes
                        jax_batch = {
                            k: jnp.asarray(v.numpy(), dtype=get_optimal_dtype(v))
                            for k, v in current_batch.items()
                        }

                        # Handle dualpipe if enabled
                        if cfg.training.get('use_dualpipe', False):
                            forward_batch, backward_batch = create_dualpipe_batch(jax_batch)
                            yield forward_batch
                            yield backward_batch
                        else:
                            # Reshape for data-parallel TPU processing
                            if num_local_devices > 1:
                                jax_batch = create_data_parallel_batch(jax_batch, num_local_devices)
                            yield jax_batch

                    except Exception as e:
                        logger.error(f"Error processing batch: {str(e)}")
                        continue

        return data_generator()

    except Exception as e:
        logger.error(f"Failed to create training dataloader: {str(e)}")
        raise

def create_val_dataloader(cfg: DictConfig) -> Iterator[Dict[str, jnp.ndarray]]:
    """Enhanced validation data loader with TPU optimizations."""
    try:
        # Handle validation data source
        if isinstance(cfg.data.val_path, str):
            val_path = get_gcs_dataset(cfg.data.val_path)
            if val_path.endswith(('.json', '.jsonl')):
                dataset = load_dataset('json', data_files=val_path)['train']
            elif val_path.endswith('.csv'):
                dataset = load_dataset('csv', data_files=val_path)['train']
            else:
                dataset = load_dataset(val_path)['validation']
        else:
            data_path = get_gcs_dataset(cfg.data.path)
            dataset = load_dataset(data_path)
            dataset = dataset['train'].train_test_split(
                test_size=cfg.data.get('val_split', 0.1),
                seed=cfg.training.get('seed', 42)
            )['test']

        # TPU-optimized preprocessing
        dataset = preprocess_dataset(
            dataset,
            cfg.model.max_seq_len,
            cfg.data.get('preprocessing', {})
        )

        # Configure for TPU evaluation
        num_local_devices = jax.local_device_count()
        global_batch_size = cfg.training.dynamic_batch_size.initial_batch_size * num_local_devices
        num_workers = min(cfg.data.get('num_workers', 4), os.cpu_count() or 4)

        loader = DataLoader(
            dataset,
            batch_size=global_batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=num_workers,
            pin_memory=True,
            prefetch_factor=2
        )

        def data_generator():
            while True:
                for batch in loader:
                    try:
                        # Convert to TPU-optimized JAX arrays
                        jax_batch = {
                            k: jnp.asarray(v.numpy(), dtype=get_optimal_dtype(v))
                            for k, v in batch.items()
                        }

                        # Reshape for data-parallel TPU processing
                        if num_local_devices > 1:
                            jax_batch = create_data_parallel_batch(jax_batch, num_local_devices)
                        yield jax_batch

                    except Exception as e:
                        logger.error(f"Error processing validation batch: {str(e)}")
                        continue

        return data_generator()

    except Exception as e:
        logger.error(f"Failed to create validation dataloader: {str(e)}")
        raise

def get_optimal_dtype(tensor) -> jnp.dtype:
    """Determine TPU-optimal dtype for tensors."""
    if tensor.dtype in [np.int32, np.int64]:
        return jnp.int32
    elif tensor.dtype in [np.float32, np.float64]:
        return jnp.bfloat16  # Use bfloat16 for TPU efficiency
    return tensor.dtype

@partial(jax.jit, static_argnums=(1,))
def preprocess_text_batch(batch: Dict[str, Any], max_length: int) -> Dict[str, Any]:
    """JIT-compiled text preprocessing for TPU."""
    return {
        k: v[:max_length] if isinstance(v, str) else v
        for k, v in batch.items()
    }

def preprocess_dataset(
    dataset: Dataset,
    max_length: int,
    preprocessing_config: Optional[Dict[str, Any]] = None
) -> Dataset:
    """TPU-optimized dataset preprocessing."""
    if preprocessing_config is None:
        preprocessing_config = {}

    def preprocess_function(examples: Dict[str, Any]) -> Dict[str, Any]:
        processed = {'text': examples['text']}

        # TPU-optimized preprocessing steps
        for step, params in preprocessing_config.items():
            if step == 'truncate':
                processed = preprocess_text_batch(processed, params.get('max_chars', max_length))
            elif step == 'clean':
                # Add TPU-optimized text cleaning here
                pass

        return processed

    try:
        # Parallel processing optimized for TPU preparation
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

@partial(jax.pmap, axis_name='batch')
def parallel_evaluate_step(state: Any, batch: Dict[str, jnp.ndarray]) -> Dict[str, jnp.ndarray]:
    """TPU-parallel evaluation step."""
    outputs = state.apply_fn({'params': state.params}, **batch, train=False)
    return jax.lax.pmean(outputs, axis_name='batch')

def evaluate(
    state: Any,
    val_loader: Iterator[Dict[str, jnp.ndarray]],
    cfg: DictConfig,
    num_batches: Optional[int] = None
) -> Dict[str, float]:
    """TPU-optimized distributed evaluation."""
    metrics_list = []
    num_local_devices = jax.local_device_count()

    # Configure evaluation batches
    if num_batches is None:
        num_batches = cfg.training.get('eval_batches', 100)
    batches_per_process = num_batches // jax.process_count()

    try:
        for _ in range(batches_per_process):
            try:
                batch = next(val_loader)
                
                # TPU-parallel evaluation
                outputs = parallel_evaluate_step(state, batch)
                
                # Extract metrics from first device
                device_outputs = jax.tree_map(lambda x: x[0], outputs)
                
                # Calculate metrics
                batch_metrics = {
                    'val_loss': float(device_outputs['loss']),
                    'val_accuracy': float(device_outputs.get('accuracy', 0.0)),
                    'val_perplexity': float(jnp.exp(jnp.minimum(device_outputs['loss'], 100)))
                }
                
                # Add any auxiliary metrics
                for key, value in device_outputs.items():
                    if key.startswith('aux_') and isinstance(value, (float, int)):
                        batch_metrics[f'val_{key}'] = float(value)
                
                metrics_list.append(batch_metrics)

            except StopIteration:
                break
            except Exception as e:
                logger.error(f"Error in evaluation step: {str(e)}")
                continue

        # Aggregate metrics across TPU devices
        if metrics_list:
            metrics = {
                k: float(np.mean([m[k] for m in metrics_list]))
                for k in metrics_list[0].keys()
            }
            
            # Add evaluation metadata
            metrics['num_processed_batches'] = len(metrics_list)
            metrics['num_devices'] = num_local_devices
            metrics['num_processes'] = jax.process_count()
            
            return metrics
        return {}

    except Exception as e:
        logger.error(f"Error in evaluation loop: {str(e)}")
        raise
