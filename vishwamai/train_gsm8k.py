import jax
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from datasets import load_dataset
from vishwamai.training import train, create_train_state
from vishwamai.model import VishwamAIModel, ModelConfig
from vishwamai.tokenizer import VishwamAITokenizer
from omegaconf import OmegaConf
import os
import logging
from safetensors.flax import save_file
from typing import Dict, Iterator
import random

logger = logging.getLogger(__name__)

class GSM8KProcessor:
    """Processor for GSM8K dataset."""
    
    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.dataset.max_length
    
    def validate_dataset_features(self, dataset):
        """Validate dataset has required features."""
        if 'question' not in dataset.features or 'answer' not in dataset.features:
            raise ValueError(f"Dataset missing required features. Found: {dataset.features}")
        
        if not dataset.features['question'].dtype == 'string' or not dataset.features['answer'].dtype == 'string':
            raise ValueError(f"Features must be string type. Found: {dataset.features}")

    def evaluate_step_accuracy(self, prediction, target):
        """Evaluate accuracy of individual solution steps."""
        pred_steps = [s.strip() for s in prediction.split('\n') if s.strip().startswith('Step:')]
        target_steps = [s.strip() for s in target.split('\n') if s.strip().startswith('Step:')]
        
        # Calculate step-level accuracy
        correct_steps = 0
        total_steps = max(len(pred_steps), len(target_steps))
        
        for i in range(min(len(pred_steps), len(target_steps))):
            if pred_steps[i] == target_steps[i]:
                correct_steps += 1
                
        return {
            'step_accuracy': correct_steps / total_steps if total_steps > 0 else 0,
            'complete_match': correct_steps == total_steps and len(pred_steps) == len(target_steps)
        }
    
    def tokenize_function(self, examples):
        """Tokenize a batch of formatted examples."""
        try:
            # Debug logging
            logger.debug(f"Processing examples of type: {type(examples)}")
            logger.debug(f"Examples attributes: {dir(examples) if hasattr(examples, '__dir__') else 'no attributes'}")
            
            # Extract fields depending on input type
            try:
                # First attempt - try direct field access
                questions = examples['question']
                answers = examples['answer']
                logger.debug("Successfully accessed fields directly")
            except Exception as e1:
                try:
                    # Second attempt - try data attribute (LazyBatch)
                    if hasattr(examples, 'data'):
                        questions = examples.data['question']
                        answers = examples.data['answer']
                        logger.debug("Successfully accessed fields via data attribute")
                    elif hasattr(examples, '_data'):
                        questions = examples._data['question']
                        answers = examples._data['answer']
                        logger.debug("Successfully accessed fields via _data attribute")
                    else:
                        raise ValueError("No data or _data attribute found")
                except Exception as e2:
                    logger.error(f"Failed both access attempts: {e1}, {e2}")
                    logger.error(f"Examples type: {type(examples)}")
                    logger.error(f"Available attributes: {dir(examples) if hasattr(examples, '__dir__') else 'no dir'}")
                    raise ValueError("Could not access question/answer fields from examples")
            
            # Validate inputs
            if not questions or not answers:
                raise ValueError("Missing questions or answers in batch")
            
            if len(questions) != len(answers):
                raise ValueError(f"Mismatched lengths: questions={len(questions)}, answers={len(answers)}")
            
            # Log batch size for debugging
            logger.debug(f"Processing batch of size: {len(questions)}")
            
            # Format examples
            formatted_texts = []
            for q, a in zip(questions, answers):
                formatted_text = (
                    f"Question: {q}\n"
                    "Let's solve this step by step:\n"
                )
                
                # Parse solution steps
                solution_parts = a.split('####')
                steps = solution_parts[0].strip().split('\n')
                final_answer = solution_parts[1].strip() if len(solution_parts) > 1 else steps[-1].strip()
                
                # Format steps
                formatted_steps = []
                for step in steps:
                    if step.strip():
                        formatted_steps.append(f"Step: {step.strip()}")
                
                formatted_text += (
                    f"{chr(10).join(formatted_steps)}\n"
                    f"Therefore, the final answer is: {final_answer}\n"
                    f"####\n{final_answer}"
                )
                formatted_texts.append(formatted_text)
            
            # Tokenize formatted texts
            tokenized = self.tokenizer(
                formatted_texts,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_attention_mask=True,
            )
            
            # Create labels for autoregressive training
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
            
        except Exception as e:
            logger.error(f"Error processing batch: {str(e)}")
            logger.error(f"Examples type: {type(examples)}")
            logger.error(f"Examples structure: {examples}")
            logger.error(f"Examples keys: {examples.keys() if hasattr(examples, 'keys') else 'no keys'}")
            raise ValueError(f"Failed to process batch: {str(e)}")
    
    def prepare_dataset(self, dataset):
        """Prepare GSM8K dataset."""
        try:
            logger.info("Starting dataset preparation...")
            logger.info(f"Dataset features: {dataset.features}")
            logger.info(f"Dataset format: {dataset.format}")
            logger.info(f"Dataset column names: {dataset.column_names}")
            
            # Validate dataset structure
            self.validate_dataset_features(dataset)
            
            # Log example for debugging
            logger.info(f"First example: {dataset[0]}")
            
            # Get preprocessing config
            preprocessing_config = self.config.dataset.preprocessing
            batch_size = preprocessing_config.batch_size
            num_proc = preprocessing_config.num_proc
            
            logger.info(f"Using batch_size={batch_size}, num_proc={num_proc}")
            
            # Process dataset with configuration settings
            tokenized_dataset = dataset.map(
                self.tokenize_function,
                batched=True,
                batch_size=batch_size,
                num_proc=num_proc,
                remove_columns=dataset.column_names,
                desc="Processing dataset",
                load_from_cache_file=False  # Disable caching during development
            )
            
            # Verify processing results
            if len(tokenized_dataset) == 0:
                raise ValueError("Dataset processing resulted in empty dataset")
            
            logger.info(f"Dataset processing complete. Size: {len(tokenized_dataset)}")
            logger.info(f"First processed example keys: {list(tokenized_dataset[0].keys())}")
            
            return tokenized_dataset
            
        except Exception as e:
            logger.error(f"Error preparing dataset: {str(e)}")
            logger.error("Dataset preparation failed", exc_info=True)
            raise
    
    def collate_fn(self, examples):
        """Collate examples into a batch."""
        batch = {
            "input_ids": np.array([example["input_ids"] for example in examples]),
            "attention_mask": np.array([example["attention_mask"] for example in examples]),
            "labels": np.array([example["labels"] for example in examples]),
        }
        return batch

def create_gsm8k_dataloader(config, split="train") -> Iterator:
    """Create data loader for GSM8K dataset."""
    # Load GSM8K dataset
    logger.info(f"Loading GSM8K dataset split: {split}")
    try:
        dataset = load_dataset("openai/gsm8k", "main", split=split)
        logger.info(f"Dataset loaded successfully. Size: {len(dataset)}")
        
        # Validate dataset schema
        required_features = ['question', 'answer']
        missing_features = [feat for feat in required_features if feat not in dataset.features]
        if missing_features:
            raise ValueError(f"Dataset missing required features: {missing_features}")
            
        for feat in required_features:
            if dataset.features[feat].dtype != 'string':
                raise ValueError(f"Feature {feat} must be string type, got {dataset.features[feat].dtype}")
                
        # Log dataset statistics
        logger.info(f"Dataset sample:\n{dataset[0]}")
        logger.info(f"Average question length: {sum(len(ex['question']) for ex in dataset) / len(dataset):.1f} chars")
        logger.info(f"Average answer length: {sum(len(ex['answer']) for ex in dataset) / len(dataset):.1f} chars")
        
    except Exception as e:
        logger.error(f"Error loading/validating GSM8K dataset: {e}")
        raise
    
    # Initialize tokenizer
    logger.info("Initializing tokenizer...")
    try:
        tokenizer = VishwamAITokenizer(
            vocab_size=config.model.vocab_size
        )
        logger.info(f"Tokenizer initialized with vocab_size: {config.model.vocab_size}")
        
        # Verify tokenizer
        test_input = "Test input string"
        test_output = tokenizer(test_input)
        logger.info("Tokenizer test successful")
        
    except Exception as e:
        logger.error(f"Failed to initialize tokenizer: {str(e)}")
        raise ValueError(f"Tokenizer initialization failed: {str(e)}")
    
    # Process dataset
    data_processor = GSM8KProcessor(tokenizer, config)
    processed_dataset = data_processor.prepare_dataset(dataset)
    
    def data_iterator():
        """Iterator that yields batches."""
        epoch = 0
        while True:
            # Shuffle at epoch start
            indices = list(range(len(processed_dataset)))
            random.shuffle(indices)
            
            # Create batches
            for i in range(0, len(indices), config.dataset.batch_size):
                batch_indices = indices[i:i + config.dataset.batch_size]
                examples = [processed_dataset[idx] for idx in batch_indices]
                yield data_processor.collate_fn(examples)
            
            epoch += 1
            logger.info(f"Completed epoch {epoch}")
    
    return data_iterator()

def save_model_safetensors(params: Dict, metrics: Dict, save_path: str):
    """Save model parameters and metrics in safetensors format."""
    # Convert parameters to numpy
    numpy_params = jax.tree_map(lambda x: np.array(x), params)
    
    # Create metadata with metrics
    metadata = {
        f"metric_{k}": str(v) for k, v in metrics.items()
    }
    metadata["timestamp"] = str(np.datetime64('now'))
    
    # Create directory if needed
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    # Save with metadata
    save_file(numpy_params, save_path, metadata=metadata)
    logger.info(f"Saved model to {save_path} with metrics: {metrics}")

def setup_tpu_cluster():
    """Set up JAX TPU cluster configuration."""
    # Get available devices
    devices = jax.devices()
    logger.info(f"Available devices: {devices}")
    
    # Create device mesh for data parallel training
    device_count = len(devices)
    device_mesh = np.array(devices).reshape(device_count)
    
    # Create mesh with data parallel sharding
    mesh = Mesh(device_mesh, ('data',))
    
    # Create sharding rules
    data_sharding = NamedSharding(mesh, P('data'))
    
    return mesh, data_sharding

def main():
    """Main training function with enhanced monitoring."""
    # Load GSM8K specific config
    config_path = "vishwamai/configs/training/gsm8k.yaml"
    config = OmegaConf.load(config_path)
    
    # Set up TPU configuration
    mesh, sharding = setup_tpu_cluster()
    
    # Initialize model and processor
    logger.info("Initializing model and GSM8K processor...")
    model_config = ModelConfig(
        vocab_size=config.model.vocab_size,
        hidden_size=config.model.hidden_size,
        num_layers=config.model.num_layers,
        num_attention_heads=config.model.num_attention_heads,
        intermediate_size=config.model.intermediate_size,
        hidden_dropout_prob=config.model.hidden_dropout_prob,
        attention_dropout_prob=config.model.attention_dropout_prob,
        max_position_embeddings=config.model.max_position_embeddings,
        layer_norm_eps=config.model.layer_norm_eps,
        use_cache=config.model.use_cache,
        pad_token_id=config.model.pad_token_id,
        bos_token_id=config.model.bos_token_id,
        eos_token_id=config.model.eos_token_id,
        tie_word_embeddings=config.model.tie_word_embeddings,
        gradient_checkpointing=config.model.gradient_checkpointing,
        use_flash_attention=config.model.use_flash_attention,
        use_rope=config.model.use_rope,
        use_alibi=config.model.use_alibi,
        use_gqa=config.model.use_gqa,
        num_key_value_heads=config.model.num_key_value_heads,
        dtype=config.model.dtype
    )
    model = VishwamAIModel(model_config)
    
    # Set up metrics tracking
    metrics_tracker = {
        'train_loss': [],
        'val_loss': [],
        'accuracy': [],
        'step_accuracy': [],
        'exact_match': []
    }
    
    # Create data loaders
    logger.info("Creating data loaders...")
    train_dataloader = create_gsm8k_dataloader(config, split="train")
    val_dataloader = create_gsm8k_dataloader(config, split="validation")
    
    # Create checkpoints directory
    checkpoint_dir = config.checkpointing.dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    logger.info("Starting training...")
    best_metrics = {
        'val_loss': float('inf'),
        'step_accuracy': 0,
        'exact_match': 0
    }
    
    with mesh:
        final_state = train(
            model,
            config,
            train_dataloader,
            val_dataloader=val_dataloader,
            num_steps=config.max_steps,
            log_every=config.monitoring.log_every_n_steps,
            eval_every=config.evaluation.eval_steps,
            checkpoint_dir=config.checkpointing.dir,
            save_checkpoint_fn=lambda state, path: save_model_safetensors(
                state.params,
                {
                    'val_loss': state.val_metrics['loss'],
                    'step_accuracy': state.val_metrics['step_accuracy'],
                    'exact_match': state.val_metrics['exact_match']
                },
                f"{path}.safetensors"
            ),
            sharding=sharding,
            metrics_tracker=metrics_tracker
        )
        
        # Save final model with all metrics
        final_metrics = {
            'val_loss': final_state.val_metrics['loss'],
            'step_accuracy': final_state.val_metrics['step_accuracy'],
            'exact_match': final_state.val_metrics['exact_match'],
            'train_loss': final_state.train_metrics['loss'],
            'total_steps': final_state.step
        }
        
        save_model_safetensors(
            final_state.params,
            final_metrics,
            os.path.join(config.checkpointing.dir, "gsm8k_final.safetensors")
        )
    
    # Log final metrics
    logger.info("Training completed!")
    for metric, value in final_metrics.items():
        if isinstance(value, float):
            logger.info(f"Final {metric}: {value:.4f}")
        else:
            logger.info(f"Final {metric}: {value}")

if __name__ == "__main__":
    # Set up logging with more detailed formatting
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    try:
        main()
    except Exception as e:
        logger.exception("Training failed with error")
        raise
