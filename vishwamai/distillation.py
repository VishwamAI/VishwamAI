"""
Knowledge Distillation Training for VishwamAI models.

This module implements comprehensive knowledge distillation training,
supporting teacher-student setups with multiple teacher models,
synthetic data generation, and advanced distillation techniques.
Uses DuckDB for experiment tracking instead of wandb.
"""

import jax
import jax.numpy as jnp
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple, Callable
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    Trainer, TrainingArguments, AutoTokenizer, AutoModel,
    pipeline, PreTrainedModel, PreTrainedTokenizer
)
from datasets import Dataset as HFDataset, load_dataset
import duckdb
import pandas as pd
from datetime import datetime
import json
import logging
from pathlib import Path
from dataclasses import dataclass, field
import time
from tqdm.auto import tqdm

try:
    from .huggingface_integration import VishwamAIForCausalLM, VishwamAIConfig, VishwamAITokenizer
    from .model import VishwamAIModel, ModelConfig
    from .training import TrainingConfig
    from .multimodal_training import MultimodalTrainer
except ImportError:
    # Fallback for when imports fail
    pass


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation training."""
    
    # Student model configuration
    student_config: dict = field(default_factory=dict)
    
    # Teacher model configuration
    teacher_model_name: str = "microsoft/DialoGPT-large"
    teacher_temperature: float = 4.0
    use_multiple_teachers: bool = False
    teacher_models: List[str] = field(default_factory=lambda: [
        "microsoft/DialoGPT-large",
        "microsoft/DialoGPT-medium",
        "facebook/opt-1.3b"
    ])
    
    # Distillation loss configuration
    distillation_alpha: float = 0.7  # Weight for distillation loss
    student_alpha: float = 0.3       # Weight for student loss
    temperature: float = 4.0         # Temperature for softmax
    use_cosine_similarity: bool = True
    use_attention_distillation: bool = True
    use_hidden_state_distillation: bool = True
    
    # Synthetic data generation
    use_synthetic_data: bool = True
    synthetic_data_ratio: float = 0.3  # Ratio of synthetic to real data
    max_synthetic_samples: int = 10000
    generation_temperature: float = 0.8
    generation_top_k: int = 50
    generation_top_p: float = 0.9
    
    # Training configuration
    output_dir: str = "./distillation_outputs"
    num_train_epochs: int = 3
    per_device_train_batch_size: int = 8
    per_device_eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 5e-5
    weight_decay: float = 0.01
    warmup_steps: int = 500
    max_steps: int = -1
    logging_steps: int = 100
    save_steps: int = 1000
    eval_steps: int = 500
    max_seq_length: int = 512
    
    # Advanced features
    use_progressive_distillation: bool = True
    progressive_stages: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"name": "easy", "max_length": 128, "epochs": 1},
        {"name": "medium", "max_length": 256, "epochs": 1},
        {"name": "hard", "max_length": 512, "epochs": 1}
    ])
    
    # Data augmentation
    use_data_augmentation: bool = True
    augmentation_techniques: List[str] = field(default_factory=lambda: [
        "paraphrase", "backtranslation", "synonym_replacement"
    ])
    
    # Multimodal distillation
    enable_multimodal_distillation: bool = False
    vision_teacher_model: str = "openai/clip-vit-base-patch32"
    audio_teacher_model: str = "facebook/wav2vec2-base"
    
    # DuckDB tracking (replaces wandb)
    use_duckdb_tracking: bool = True
    duckdb_path: str = "./experiments.db"
    experiment_name: Optional[str] = None
    log_predictions: bool = True
    save_best_model: bool = True


class DuckDBDistillationTracker:
    """DuckDB-based experiment tracking for distillation experiments."""
    
    def __init__(self, db_path: str, experiment_name: str = None):
        """Initialize DuckDB distillation tracker."""
        self.db_path = db_path
        self.experiment_name = experiment_name or f"distillation_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.conn = duckdb.connect(db_path)
        self.run_id = f"{self.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._setup_distillation_tables()
        
    def _setup_distillation_tables(self):
        """Create distillation-specific tables."""
        # Base experiment tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS distillation_experiments (
                run_id TEXT PRIMARY KEY,
                experiment_name TEXT,
                start_time TIMESTAMP,
                end_time TIMESTAMP,
                status TEXT,
                student_config JSON,
                teacher_models JSON,
                distillation_config JSON,
                hardware_info JSON,
                final_metrics JSON
            )
        """)
        
        # Training metrics with distillation-specific fields
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS distillation_metrics (
                run_id TEXT,
                step INTEGER,
                epoch INTEGER,
                metric_name TEXT,
                metric_value DOUBLE,
                metric_type TEXT,
                teacher_model TEXT,
                temperature DOUBLE,
                timestamp TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES distillation_experiments(run_id)
            )
        """)
        
        # Knowledge transfer analysis
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS knowledge_transfer (
                run_id TEXT,
                step INTEGER,
                epoch INTEGER,
                layer_idx INTEGER,
                transfer_type TEXT,
                kl_divergence DOUBLE,
                attention_transfer_loss DOUBLE,
                hidden_state_similarity DOUBLE,
                timestamp TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES distillation_experiments(run_id)
            )
        """)
        
        # Synthetic data quality tracking
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS synthetic_data_quality (
                run_id TEXT,
                sample_id TEXT,
                prompt TEXT,
                generated_text TEXT,
                teacher_model TEXT,
                quality_score DOUBLE,
                perplexity DOUBLE,
                semantic_similarity DOUBLE,
                generation_params JSON,
                timestamp TIMESTAMP,
                FOREIGN KEY (run_id) REFERENCES distillation_experiments(run_id)
            )
        """)
    
    def start_experiment(self, config: dict, hardware_info: dict):
        """Start a new distillation experiment."""
        self.conn.execute("""
            INSERT INTO distillation_experiments 
            (run_id, experiment_name, start_time, status, student_config, teacher_models, distillation_config, hardware_info)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            self.run_id,
            self.experiment_name,
            datetime.now(),
            "running",
            json.dumps(config.get('student_config', {})),
            json.dumps(config.get('teacher_models', [])),
            json.dumps({
                'temperature': config.get('temperature', 4.0),
                'distillation_alpha': config.get('distillation_alpha', 0.7),
                'use_attention_distillation': config.get('use_attention_distillation', False),
                'use_hidden_state_distillation': config.get('use_hidden_state_distillation', False)
            }),
            json.dumps(hardware_info)
        ])
    
    def log_distillation_metrics(self, metrics: dict, step: int, epoch: int, teacher_model: str = None, temperature: float = None):
        """Log distillation-specific metrics."""
        timestamp = datetime.now()
        for name, value in metrics.items():
            if isinstance(value, (int, float)):
                self.conn.execute("""
                    INSERT INTO distillation_metrics 
                    (run_id, step, epoch, metric_name, metric_value, metric_type, teacher_model, temperature, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, [self.run_id, step, epoch, name, float(value), "distillation", teacher_model, temperature, timestamp])
    
    def log_knowledge_transfer(self, step: int, epoch: int, layer_idx: int, transfer_metrics: dict):
        """Log knowledge transfer analysis."""
        self.conn.execute("""
            INSERT INTO knowledge_transfer 
            (run_id, step, epoch, layer_idx, transfer_type, kl_divergence, attention_transfer_loss, hidden_state_similarity, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, [
            self.run_id, step, epoch, layer_idx,
            transfer_metrics.get('transfer_type', 'standard'),
            transfer_metrics.get('kl_divergence', 0.0),
            transfer_metrics.get('attention_transfer_loss', 0.0),
            transfer_metrics.get('hidden_state_similarity', 0.0),
            datetime.now()
        ])
    
    def log_synthetic_data_quality(self, samples: list):
        """Log synthetic data quality metrics."""
        timestamp = datetime.now()
        for i, sample in enumerate(samples):
            sample_id = f"{self.run_id}_synthetic_{i}"
            self.conn.execute("""
                INSERT INTO synthetic_data_quality 
                (run_id, sample_id, prompt, generated_text, teacher_model, quality_score, 
                 perplexity, semantic_similarity, generation_params, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                self.run_id, sample_id,
                sample.get("prompt", ""),
                sample.get("generated_text", ""),
                sample.get("teacher_model", "unknown"),
                sample.get("quality_score", 0.0),
                sample.get("perplexity", 0.0),
                sample.get("semantic_similarity", 0.0),
                json.dumps(sample.get("generation_params", {})),
                timestamp
            ])
    
    def finish_experiment(self, final_metrics: dict, status: str = "completed"):
        """Mark experiment as finished."""
        self.conn.execute("""
            UPDATE distillation_experiments 
            SET end_time = ?, status = ?, final_metrics = ?
            WHERE run_id = ?
        """, [datetime.now(), status, json.dumps(final_metrics), self.run_id])
    
    def export_to_csv(self, output_dir: str):
        """Export experiment data to CSV files."""
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Export metrics
        metrics_df = self.conn.execute("""
            SELECT * FROM distillation_metrics WHERE run_id = ?
        """, [self.run_id]).df()
        if not metrics_df.empty:
            metrics_df.to_csv(f"{output_dir}/distillation_metrics.csv", index=False)
        
        # Export knowledge transfer
        transfer_df = self.conn.execute("""
            SELECT * FROM knowledge_transfer WHERE run_id = ?
        """, [self.run_id]).df()
        if not transfer_df.empty:
            transfer_df.to_csv(f"{output_dir}/knowledge_transfer.csv", index=False)
        
        # Export synthetic data quality
        synthetic_df = self.conn.execute("""
            SELECT * FROM synthetic_data_quality WHERE run_id = ?
        """, [self.run_id]).df()
        if not synthetic_df.empty:
            synthetic_df.to_csv(f"{output_dir}/synthetic_data_quality.csv", index=False)
    
    def close(self):
        """Close database connection."""
        if self.conn:
            self.conn.close()


class TeacherEnsemble:
    """Ensemble of teacher models for knowledge distillation."""
    
    def __init__(
        self,
        teacher_names: List[str],
        tokenizer: PreTrainedTokenizer,
        device: str = "auto"
    ):
        self.teacher_names = teacher_names
        self.tokenizer = tokenizer
        self.device = device
        self.teachers = {}
        
        # Load teacher models
        for name in teacher_names:
            try:
                model = AutoModel.from_pretrained(name, torch_dtype=torch.float16)
                if device != "cpu":
                    model = model.to(device)
                model.eval()
                self.teachers[name] = model
                logging.info(f"Loaded teacher model: {name}")
            except Exception as e:
                logging.warning(f"Failed to load teacher {name}: {e}")
    
    def get_teacher_outputs(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_hidden_states: bool = True,
        return_attentions: bool = True
    ) -> Dict[str, Dict[str, torch.Tensor]]:
        """Get outputs from all teacher models."""
        
        teacher_outputs = {}
        
        for name, teacher in self.teachers.items():
            with torch.no_grad():
                outputs = teacher(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    output_hidden_states=return_hidden_states,
                    output_attentions=return_attentions,
                    return_dict=True
                )
                
                teacher_outputs[name] = {
                    "logits": outputs.logits if hasattr(outputs, "logits") else None,
                    "hidden_states": outputs.hidden_states if return_hidden_states else None,
                    "attentions": outputs.attentions if return_attentions else None
                }
        
        return teacher_outputs
    
    def generate_synthetic_data(
        self,
        prompts: List[str],
        max_length: int = 512,
        temperature: float = 0.8,
        num_samples_per_prompt: int = 2
    ) -> List[Dict[str, str]]:
        """Generate synthetic training data using teacher models."""
        
        synthetic_data = []
        
        for prompt in tqdm(prompts, desc="Generating synthetic data"):
            for teacher_name, teacher in self.teachers.items():
                try:
                    # Use generation pipeline
                    generator = pipeline(
                        "text-generation",
                        model=teacher,
                        tokenizer=self.tokenizer,
                        device=self.device
                    )
                    
                    outputs = generator(
                        prompt,
                        max_length=max_length,
                        temperature=temperature,
                        num_return_sequences=num_samples_per_prompt,
                        do_sample=True,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    
                    for output in outputs:
                        synthetic_data.append({
                            "prompt": prompt,
                            "generated_text": output["generated_text"],
                            "teacher": teacher_name,
                            "type": "synthetic"
                        })
                
                except Exception as e:
                    logging.warning(f"Failed to generate with {teacher_name}: {e}")
        
        return synthetic_data


class DistillationDataset(Dataset):
    """Dataset for knowledge distillation training."""
    
    def __init__(
        self,
        texts: List[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        synthetic_data: Optional[List[Dict[str, str]]] = None
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.synthetic_data = synthetic_data or []
        
        # Combine real and synthetic data
        self.all_texts = texts + [item["generated_text"] for item in self.synthetic_data]
    
    def __len__(self):
        return len(self.all_texts)
    
    def __getitem__(self, idx):
        text = self.all_texts[idx]
        
        # Tokenize
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt"
        )
        
        # Create labels (same as input_ids for language modeling)
        labels = encoding["input_ids"].clone()
        labels[labels == self.tokenizer.pad_token_id] = -100
        
        return {
            "input_ids": encoding["input_ids"].squeeze(),
            "attention_mask": encoding["attention_mask"].squeeze(),
            "labels": labels.squeeze(),
            "is_synthetic": idx >= len(self.texts)
        }


class DistillationLoss:
    """Comprehensive loss function for knowledge distillation."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        self.kl_div = torch.nn.KLDivLoss(reduction="batchmean")
        self.cross_entropy = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()
        self.cosine_similarity = torch.nn.CosineSimilarity(dim=-1)
    
    def compute_kl_divergence_loss(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
        temperature: float
    ) -> torch.Tensor:
        """Compute KL divergence loss between student and teacher logits."""
        
        student_probs = torch.nn.functional.log_softmax(
            student_logits / temperature, dim=-1
        )
        teacher_probs = torch.nn.functional.softmax(
            teacher_logits / temperature, dim=-1
        )
        
        return self.kl_div(student_probs, teacher_probs) * (temperature ** 2)
    
    def compute_attention_loss(
        self,
        student_attentions: Tuple[torch.Tensor],
        teacher_attentions: Tuple[torch.Tensor]
    ) -> torch.Tensor:
        """Compute attention transfer loss."""
        
        attention_loss = 0.0
        num_layers = min(len(student_attentions), len(teacher_attentions))
        
        for i in range(num_layers):
            student_attn = student_attentions[i]
            teacher_attn = teacher_attentions[i]
            
            # Average across heads
            student_attn = student_attn.mean(dim=1)
            teacher_attn = teacher_attn.mean(dim=1)
            
            # MSE loss between attention matrices
            attention_loss += self.mse_loss(student_attn, teacher_attn)
        
        return attention_loss / num_layers
    
    def compute_hidden_state_loss(
        self,
        student_hidden_states: Tuple[torch.Tensor],
        teacher_hidden_states: Tuple[torch.Tensor]
    ) -> torch.Tensor:
        """Compute hidden state alignment loss."""
        
        hidden_loss = 0.0
        num_layers = min(len(student_hidden_states), len(teacher_hidden_states))
        
        for i in range(num_layers):
            student_hidden = student_hidden_states[i]
            teacher_hidden = teacher_hidden_states[i]
            
            # Use cosine similarity or MSE
            if self.config.use_cosine_similarity:
                similarity = self.cosine_similarity(student_hidden, teacher_hidden)
                hidden_loss += (1 - similarity.mean())
            else:
                hidden_loss += self.mse_loss(student_hidden, teacher_hidden)
        
        return hidden_loss / num_layers
    
    def compute_total_loss(
        self,
        student_outputs: Dict[str, torch.Tensor],
        teacher_outputs: Dict[str, Dict[str, torch.Tensor]],
        labels: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """Compute total distillation loss."""
        
        losses = {}
        total_loss = 0.0
        
        # Student language modeling loss
        if "logits" in student_outputs:
            student_loss = self.cross_entropy(
                student_outputs["logits"].view(-1, student_outputs["logits"].size(-1)),
                labels.view(-1)
            )
            losses["student_loss"] = student_loss
            total_loss += self.config.student_alpha * student_loss
        
        # Distillation losses (average across teachers)
        distillation_losses = []
        for teacher_name, teacher_output in teacher_outputs.items():
            if teacher_output["logits"] is not None:
                kl_loss = self.compute_kl_divergence_loss(
                    student_outputs["logits"],
                    teacher_output["logits"],
                    self.config.temperature
                )
                distillation_losses.append(kl_loss)
                losses[f"kl_loss_{teacher_name}"] = kl_loss
        
        if distillation_losses:
            avg_distillation_loss = torch.stack(distillation_losses).mean()
            losses["distillation_loss"] = avg_distillation_loss
            total_loss += self.config.distillation_alpha * avg_distillation_loss
        
        # Attention distillation
        if (self.config.use_attention_distillation and 
            "attentions" in student_outputs):
            
            attention_losses = []
            for teacher_name, teacher_output in teacher_outputs.items():
                if teacher_output["attentions"] is not None:
                    attn_loss = self.compute_attention_loss(
                        student_outputs["attentions"],
                        teacher_output["attentions"]
                    )
                    attention_losses.append(attn_loss)
            
            if attention_losses:
                avg_attention_loss = torch.stack(attention_losses).mean()
                losses["attention_loss"] = avg_attention_loss
                total_loss += 0.1 * avg_attention_loss
        
        # Hidden state distillation
        if (self.config.use_hidden_state_distillation and 
            "hidden_states" in student_outputs):
            
            hidden_losses = []
            for teacher_name, teacher_output in teacher_outputs.items():
                if teacher_output["hidden_states"] is not None:
                    hidden_loss = self.compute_hidden_state_loss(
                        student_outputs["hidden_states"],
                        teacher_output["hidden_states"]
                    )
                    hidden_losses.append(hidden_loss)
            
            if hidden_losses:
                avg_hidden_loss = torch.stack(hidden_losses).mean()
                losses["hidden_loss"] = avg_hidden_loss
                total_loss += 0.1 * avg_hidden_loss
        
        losses["total_loss"] = total_loss
        return losses


class DistillationTrainer(Trainer):
    """Custom trainer for knowledge distillation with DuckDB tracking."""
    
    def __init__(
        self,
        model,
        teacher_ensemble: TeacherEnsemble,
        config: DistillationConfig,
        tracker: Optional[DuckDBDistillationTracker] = None,
        **kwargs
    ):
        self.teacher_ensemble = teacher_ensemble
        self.distillation_config = config
        self.distillation_loss = DistillationLoss(config)
        self.tracker = tracker
        
        # Initialize Trainer
        training_args = TrainingArguments(
            output_dir=config.output_dir,
            num_train_epochs=config.num_train_epochs,
            per_device_train_batch_size=config.per_device_train_batch_size,
            per_device_eval_batch_size=config.per_device_eval_batch_size,
            gradient_accumulation_steps=config.gradient_accumulation_steps,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            warmup_steps=config.warmup_steps,
            max_steps=config.max_steps,
            logging_steps=config.logging_steps,
            save_steps=config.save_steps,
            eval_steps=config.eval_steps,
            save_total_limit=3,
            load_best_model_at_end=config.save_best_model,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            remove_unused_columns=False,
        )
        
        super().__init__(
            model=model,
            args=training_args,
            **kwargs
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Compute distillation loss."""
        
        labels = inputs.get("labels")
        
        # Student forward pass
        student_outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            return_dict=True,
            output_hidden_states=self.distillation_config.use_hidden_state_distillation,
            output_attentions=self.distillation_config.use_attention_distillation
        )
        
        # Teacher forward pass
        teacher_outputs = self.teacher_ensemble.get_teacher_outputs(
            inputs["input_ids"],
            inputs["attention_mask"],
            return_hidden_states=self.distillation_config.use_hidden_state_distillation,
            return_attentions=self.distillation_config.use_attention_distillation
        )
        
        # Compute losses
        loss_dict = self.distillation_loss.compute_total_loss(
            {
                "logits": student_outputs.logits,
                "hidden_states": getattr(student_outputs, "hidden_states", None),
                "attentions": getattr(student_outputs, "attentions", None)
            },
            teacher_outputs,
            labels
        )
        
        # Log individual losses to DuckDB
        if self.tracker and self.state.is_world_process_zero:
            step = self.state.global_step
            epoch = self.state.epoch
            
            # Log metrics to DuckDB
            metrics = {key: value.item() for key, value in loss_dict.items() if key != "total_loss"}
            self.tracker.log_distillation_metrics(
                metrics, step, epoch, 
                temperature=self.distillation_config.temperature
            )
        
        loss = loss_dict["total_loss"]
        
        return (loss, student_outputs) if return_outputs else loss


def create_distillation_dataset(
    dataset_name: str = "wikitext-2-raw-v1",
    split: str = "train",
    max_samples: int = 10000,
    config: Optional[DistillationConfig] = None
) -> Tuple[List[str], Optional[List[Dict[str, str]]]]:
    """Create dataset for distillation training."""
    
    # Load base dataset
    dataset = load_dataset(dataset_name, split=split)
    
    if max_samples > 0:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    texts = [item["text"] for item in dataset if len(item["text"].strip()) > 10]
    
    synthetic_data = None
    if config and config.use_synthetic_data:
        # Generate synthetic data using teacher models
        logging.info("Generating synthetic data...")
        
        # Use a subset of texts as prompts
        num_prompts = min(100, len(texts) // 10)
        prompts = texts[:num_prompts]
        
        # Create teacher ensemble for generation
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        teacher_ensemble = TeacherEnsemble(
            config.teacher_models if config.use_multiple_teachers else [config.teacher_model_name],
            tokenizer
        )
        
        synthetic_data = teacher_ensemble.generate_synthetic_data(
            prompts,
            max_length=config.max_seq_length,
            temperature=config.generation_temperature,
            num_samples_per_prompt=2
        )
        
        # Limit synthetic data
        if len(synthetic_data) > config.max_synthetic_samples:
            synthetic_data = synthetic_data[:config.max_synthetic_samples]
        
        logging.info(f"Generated {len(synthetic_data)} synthetic samples")
    
    return texts, synthetic_data


def train_with_distillation(
    config: DistillationConfig,
    train_dataset: Optional[DistillationDataset] = None,
    eval_dataset: Optional[DistillationDataset] = None,
    tokenizer: Optional[PreTrainedTokenizer] = None
):
    """Main function to train VishwamAI with knowledge distillation using DuckDB tracking."""
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize DuckDB tracker if enabled
    tracker = None
    if config.use_duckdb_tracking:
        tracker = DuckDBDistillationTracker(
            config.duckdb_path, 
            config.experiment_name
        )
        
        # Start experiment tracking
        hardware_info = {
            "device": str(torch.device("cuda" if torch.cuda.is_available() else "cpu")),
            "torch_version": torch.__version__,
        }
        if torch.cuda.is_available():
            hardware_info["cuda_version"] = torch.version.cuda
            hardware_info["gpu_count"] = torch.cuda.device_count()
        
        tracker.start_experiment(config.__dict__, hardware_info)
    
    # Setup tokenizer
    if tokenizer is None:
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
    
    # Create datasets if not provided
    if train_dataset is None:
        logger.info("Creating training dataset...")
        texts, synthetic_data = create_distillation_dataset(
            max_samples=10000,
            config=config
        )
        train_dataset = DistillationDataset(
            texts[:8000],
            tokenizer,
            config.max_seq_length,
            synthetic_data
        )
        
        eval_dataset = DistillationDataset(
            texts[8000:],
            tokenizer,
            config.max_seq_length
        )
        
        # Log synthetic data quality if tracker is available
        if tracker and synthetic_data:
            tracker.log_synthetic_data_quality(synthetic_data[:100])  # Log first 100 samples
    
    # Initialize student model (simplified version)
    logger.info("Initializing student model...")
    try:
        from .model import VishwamAIModel
        student_model = VishwamAIModel(config.student_config)
    except ImportError:
        # Fallback to a simple model for testing
        from transformers import GPT2LMHeadModel, GPT2Config
        student_config = GPT2Config(vocab_size=tokenizer.vocab_size, n_layer=6, n_head=8, n_embd=512)
        student_model = GPT2LMHeadModel(student_config)
    
    # Initialize teacher ensemble
    logger.info("Loading teacher models...")
    teacher_ensemble = TeacherEnsemble(
        config.teacher_models if config.use_multiple_teachers else [config.teacher_model_name],
        tokenizer
    )
    
    # Create trainer
    trainer = DistillationTrainer(
        model=student_model,
        teacher_ensemble=teacher_ensemble,
        config=config,
        tracker=tracker,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer
    )
    
    try:
        # Progressive training if enabled
        if config.use_progressive_distillation:
            logger.info("Starting progressive distillation training...")
            
            for stage in config.progressive_stages:
                logger.info(f"Training stage: {stage['name']}")
                
                # Update max length for this stage
                stage_train_dataset = DistillationDataset(
                    [item for item in train_dataset.all_texts],
                    tokenizer,
                    stage["max_length"]
                )
                
                trainer.train_dataset = stage_train_dataset
                trainer.args.num_train_epochs = stage["epochs"]
                
                # Train for this stage
                trainer.train()
        else:
            # Standard training
            logger.info("Starting distillation training...")
            trainer.train()
        
        # Save final model
        logger.info("Saving trained model...")
        trainer.save_model()
        
        # Evaluate final model
        final_metrics = {}
        if eval_dataset:
            eval_results = trainer.evaluate()
            final_metrics.update(eval_results)
        
        # Finish experiment tracking
        if tracker:
            tracker.finish_experiment(final_metrics)
            logger.info(f"Experiment tracking completed. Results saved to {config.duckdb_path}")
    
    except Exception as e:
        logger.error(f"Training failed: {e}")
        if tracker:
            tracker.finish_experiment({}, status="failed")
        raise
    
    finally:
        # Close tracker
        if tracker:
            tracker.close()
    
    return student_model


def evaluate_distilled_model(
    model,
    tokenizer: PreTrainedTokenizer,
    eval_dataset: DistillationDataset,
    metrics: List[str] = ["perplexity", "sample_generation"]
) -> Dict[str, float]:
    """Evaluate the distilled model on various metrics."""
    
    results = {}
    
    # Compute perplexity
    if "perplexity" in metrics:
        total_loss = 0
        total_tokens = 0
        
        model.eval()
        with torch.no_grad():
            for batch in DataLoader(eval_dataset, batch_size=8):
                try:
                    outputs = model(**batch)
                    loss = outputs.loss
                    total_loss += loss.item() * batch["input_ids"].size(0)
                    total_tokens += batch["input_ids"].size(0)
                except Exception as e:
                    logging.warning(f"Error computing perplexity batch: {e}")
                    continue
        
        if total_tokens > 0:
            perplexity = torch.exp(torch.tensor(total_loss / total_tokens))
            results["perplexity"] = perplexity.item()
    
    # Generate sample outputs for qualitative evaluation
    if "sample_generation" in metrics:
        sample_inputs = [
            "The future of artificial intelligence is",
            "Climate change is a global challenge that",
            "In the field of medicine, recent advances"
        ]
        
        results["sample_generations"] = []
        for input_text in sample_inputs:
            try:
                tokens = tokenizer(input_text, return_tensors="pt")
                with torch.no_grad():
                    generated = model.generate(
                        tokens["input_ids"],
                        max_length=100,
                        temperature=0.7,
                        do_sample=True,
                        pad_token_id=tokenizer.eos_token_id
                    )
                
                generated_text = tokenizer.decode(generated[0], skip_special_tokens=True)
                results["sample_generations"].append({
                    "input": input_text,
                    "output": generated_text
                })
            except Exception as e:
                logging.warning(f"Error generating sample for '{input_text}': {e}")
    
    return results


# Example usage and testing functions
def run_distillation_experiment(config_path: Optional[str] = None):
    """Run a complete distillation experiment."""
    
    # Load or create config
    if config_path and Path(config_path).exists():
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
        config = DistillationConfig(**config_dict)
    else:
        config = DistillationConfig(
            use_duckdb_tracking=True,
            duckdb_path="./distillation_experiments.db",
            experiment_name="test_distillation",
            num_train_epochs=1,
            max_steps=100,
            per_device_train_batch_size=4
        )
    
    # Run training
    model = train_with_distillation(config)
    
    # Evaluate
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    texts, _ = create_distillation_dataset(max_samples=100, config=config)
    eval_dataset = DistillationDataset(texts, tokenizer, config.max_seq_length)
    
    results = evaluate_distilled_model(model, tokenizer, eval_dataset)
    
    print("Evaluation Results:")
    for key, value in results.items():
        if key != "sample_generations":
            print(f"{key}: {value}")
    
    return model, results


if __name__ == "__main__":
    # Run a simple test
    run_distillation_experiment()
