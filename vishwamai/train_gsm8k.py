"""
Enhanced GSM8K training module with deep ToT integration.
"""

import jax
import jax.numpy as jnp
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import numpy as np
from datasets import load_dataset
from vishwamai.training import train, create_train_state
from vishwamai.model import VishwamAIModel, ModelConfig
from vishwamai.tokenizer import VishwamAITokenizer
from vishwamai.error_correction import ErrorCorrectionTrainer
from vishwamai.tot import TreeOfThoughts, Thought
from omegaconf import OmegaConf
import os
import logging
from safetensors.flax import save_file
from typing import Dict, Iterator, Optional
import random

logger = logging.getLogger(__name__)

class GSM8KProcessor:
    """Processor for GSM8K dataset with ToT-enhanced reasoning."""
    
    def __init__(self, tokenizer: VishwamAITokenizer, config):
        self.tokenizer = tokenizer
        self.config = config
        self.max_length = config.dataset.max_length
    
    def validate_dataset_features(self, dataset):
        """Validate dataset has required features."""
        required = ['question', 'answer']
        if not all(feat in dataset.features for feat in required):
            raise ValueError(f"Dataset missing required features: {dataset.features}")
        if not all(dataset.features[feat].dtype == 'string' for feat in required):
            raise ValueError(f"Features must be strings: {dataset.features}")

    def evaluate_step_accuracy(self, prediction: str, target: str) -> Dict[str, float]:
        """Evaluate step-level accuracy and exact match with ToT outputs."""
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
        """Tokenize GSM8K examples with ToT-compatible formatting."""
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
                tokenized["error_weights"][:, token_pos:] = 2.0  # Emphasize answer
            
            return tokenized
        except Exception as e:
            logger.error(f"Tokenization error: {str(e)}")
            raise
    
    def prepare_dataset(self, dataset):
        """Prepare GSM8K dataset with ToT support."""
        self.validate_dataset_features(dataset)
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            batch_size=self.config.dataset.preprocessing.batch_size,
            num_proc=self.config.dataset.preprocessing.num_proc,
            remove_columns=dataset.column_names,
            desc="Tokenizing GSM8K dataset"
        )
        return tokenized_dataset
    
    def collate_fn(self, examples):
        """Collate examples into a batch with error weights."""
        batch = {
            "input_ids": np.array([ex["input_ids"] for ex in examples]),
            "attention_mask": np.array([ex["attention_mask"] for ex in examples]),
            "labels": np.array([ex["labels"] for ex in examples]),
            "error_weights": np.array([ex["error_weights"] for ex in examples])
        }
        return batch

def create_gsm8k_dataloader(config, split="train") -> Iterator:
    """Create GSM8K data loader with ToT-enhanced batches."""
    dataset = load_dataset("openai/gsm8k", "main", split=split)
    processor = GSM8KProcessor(config.training.tokenizer, config)
    processed_dataset = processor.prepare_dataset(dataset)
    
    def data_iterator():
        """Infinite iterator with curriculum shuffling."""
        epoch = 0
        while True:
            indices = list(range(len(processed_dataset)))
            if config.training.get('use_curriculum', True) and epoch < config.training.max_steps // config.training.eval_every:
                indices.sort(key=lambda i: len(processed_dataset[i]['input_ids']))
            random.shuffle(indices)
            
            for i in range(0, len(indices), config.dataset.batch_size):
                batch_indices = indices[i:i + config.dataset.batch_size]
                examples = [processed_dataset[idx] for idx in batch_indices]
                yield processor.collate_fn(examples)
            epoch += 1
            logger.info(f"Epoch {epoch} completed")
    
    return data_iterator()

def save_model_safetensors(params: Dict, metrics: Dict, save_path: str):
    """Save model parameters and metrics in safetensors format."""
    numpy_params = jax.tree_map(np.asarray, params)
    metadata = {f"metric_{k}": str(v) for k, v in metrics.items()}
    metadata["timestamp"] = str(np.datetime64('now'))
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_file(numpy_params, save_path, metadata=metadata)
    logger.info(f"Saved model to {save_path} with metrics: {metrics}")

def setup_tpu_cluster():
    """Set up JAX TPU cluster with enhanced sharding."""
    devices = jax.devices()
    logger.info(f"Available devices: {devices}")
    
    device_count = len(devices)
    device_mesh = np.array(devices).reshape((device_count // 2, 2))
    
    mesh = Mesh(device_mesh, ('data', 'model'))
    sharding = NamedSharding(mesh, P('data', 'model'))
    
    return mesh, sharding

def main(config_path: str = "vishwamai/configs/training/gsm8k.yaml"):
    """Enhanced GSM8K training with full ToT and error correction integration."""
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Load configuration
    config = OmegaConf.load(config_path)
    model_config = ModelConfig(**config.model)
    model = VishwamAIModel(model_config)
    
    # Initialize tokenizer
    tokenizer = VishwamAITokenizer(vocab_size=config.model.vocab_size)
    tokenizer.train([config.dataset.dataset_path], "tokenizer_output")
    config.training.tokenizer = tokenizer
    
    # Initialize ToT
    tot = TreeOfThoughts(
        transformer=model,
        tokenizer=tokenizer,
        max_thoughts=config.training.get('tot_max_thoughts', 5),
        max_depth=config.training.get('tot_max_depth', 3),
        beam_width=config.training.get('tot_beam_width', 5)
    )
    model.tot_model = tot  # Attach ToT to model for training
    
    # Initialize Error Correction
    error_trainer = ErrorCorrectionTrainer(
        config=config,
        transformer=model,
        tokenizer=tokenizer,
        use_tot=config.training.get('use_tot', True),
        use_mod=config.model.get('use_mod', True),
        history_size=config.training.get('error_history_size', 100),
        threshold_percentile=config.training.get('error_threshold_percentile', 85.0)
    )
    
    # Setup TPU cluster
    mesh, sharding = setup_tpu_cluster()
    
    # Create data loaders
    train_dataloader = create_gsm8k_dataloader(config, "train")
    val_dataloader = create_gsm8k_dataloader(config, "test")
    
    # Create checkpoints directory
    checkpoint_dir = config.checkpointing.dir
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Metrics tracker
    processor = GSM8KProcessor(tokenizer, config)
    metrics_tracker = {
        'train_loss': [], 'val_loss': [], 'step_accuracy': [], 'exact_match': [], 'answer_match': [], 'tot_score': []
    }
    
    with mesh:
        rng = jax.random.PRNGKey(config.training.seed)
        state = create_train_state(model, config, rng)
        
        # Initialize error correction params
        dummy_input = jnp.ones((1, config.dataset.max_length, config.model.hidden_size))
        error_trainer.init_params(rng, dummy_input)
        
        final_state = train(
            model,
            config,
            train_dataloader,
            val_dataloader=val_dataloader,
            num_steps=config.training.max_steps,
            log_every=config.monitoring.log_every_n_steps,
            eval_every=config.evaluation.eval_steps,
            checkpoint_dir=checkpoint_dir,
            accum_steps=config.training.get('accum_steps', 1)
        )
        
        # Enhanced final evaluation with ToT and error correction
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
            
            # Apply error correction
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