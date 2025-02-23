"""Test suite for HuggingFace training integration."""

import os
import unittest
import torch
import torch_xla.core.xla_model as xm
from datasets import load_dataset
from transformers import AutoTokenizer

from vishwamai.model.transformer import create_transformer_model, get_pretrained_config
from vishwamai.training.distributed import tpu_utils
from vishwamai.data.dataset.implementations.gsm8k import GSM8KDataset
from vishwamai.utils.logging import get_logger

logger = get_logger(__name__)

class TestGSM8KTraining(unittest.TestCase):
    """Test GSM8K training functionality."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test environment."""
        # Load TPU configuration
        cls.tpu_config = tpu_utils.load_tpu_config("vishwamai/configs/tpu_config.yaml")
        
        # Initialize TPU
        cls.device = xm.xla_device()
        cls.rank = xm.get_ordinal()
        cls.world_size = xm.xrt_world_size()
        
        # Load small subset of GSM8K for testing
        cls.dataset = load_dataset(
            "openai/gsm8k",
            "main",
            split="train[:100]"  # Small subset for testing
        )
        
        # Initialize tokenizer
        cls.tokenizer = AutoTokenizer.from_pretrained("gpt2", pad_token="<pad>")
        cls.tokenizer.add_special_tokens({
            'sep_token': '<sep>',
            'cls_token': '<cls>'
        })
        
        # Create model configuration
        cls.model_config = get_pretrained_config(
            model_size="small",  # Use small model for testing
            model_type="moe_mla_transformer"
        )
        
    def test_model_creation(self):
        """Test model initialization."""
        model = create_transformer_model(self.model_config)
        model = model.to(self.device)
        
        self.assertIsNotNone(model)
        self.assertEqual(model.config.hidden_size, self.model_config.hidden_size)
        
    def test_data_loading(self):
        """Test GSM8K dataset loading."""
        dataset = GSM8KDataset(
            self.dataset,
            tokenizer=self.tokenizer,
            max_length=128  # Small for testing
        )
        
        self.assertEqual(len(dataset), 100)  # Test subset size
        
        # Check sample format
        sample = dataset[0]
        self.assertIn("input_ids", sample)
        self.assertIn("attention_mask", sample)
        self.assertIn("labels", sample)
        
    def test_tpu_optimization(self):
        """Test TPU-specific optimizations."""
        model = create_transformer_model(self.model_config)
        model = model.to(self.device)
        
        # Apply TPU optimizations
        optimized_model = tpu_utils.optimize_tpu_execution(model, self.tpu_config)
        self.assertIsNotNone(optimized_model)
        
    def test_training_step(self):
        """Test single training step."""
        # Create model and move to TPU
        model = create_transformer_model(self.model_config)
        model = model.to(self.device)
        
        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        
        # Create sample batch
        batch = {
            "input_ids": torch.randint(0, 1000, (2, 64)).to(self.device),
            "attention_mask": torch.ones(2, 64).to(self.device),
            "labels": torch.randint(0, 1000, (2, 64)).to(self.device)
        }
        
        # Training step
        model.train()
        outputs = model(**batch)
        loss = outputs["loss"]
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        xm.mark_step()
        
        self.assertIsNotNone(loss)
        self.assertTrue(torch.isfinite(loss))
        
    def test_model_saving(self):
        """Test model checkpoint saving and loading."""
        model = create_transformer_model(self.model_config)
        model = model.to(self.device)
        
        # Save checkpoint
        checkpoint_path = "test_checkpoint.pt"
        if self.rank == 0:
            tpu_utils.save_checkpoint(
                model=model,
                optimizer=None,
                filepath=checkpoint_path,
                step=0,
                tpu_config=self.tpu_config
            )
        
        # Load checkpoint
        loaded_model = create_transformer_model(self.model_config)
        loaded_model, _, _, _ = tpu_utils.load_checkpoint(
            model=loaded_model,
            optimizer=None,
            filepath=checkpoint_path,
            device=self.device
        )
        
        self.assertIsNotNone(loaded_model)
        
        # Cleanup
        if os.path.exists(checkpoint_path):
            os.remove(checkpoint_path)
            
    def test_expert_sharding(self):
        """Test MoE expert sharding across TPU cores."""
        from vishwamai.training.distributed.expert_sharding import ExpertShardingManager
        
        sharding_manager = ExpertShardingManager(
            num_experts=8,
            num_tpu_cores=self.world_size,
            strategy="axis_0",
            device=self.device
        )
        
        # Create model with experts
        model = create_transformer_model(self.model_config)
        model = model.to(self.device)
        
        # Get local experts
        local_experts = sharding_manager.local_experts
        self.assertGreater(len(local_experts), 0)
        
    def test_generation(self):
        """Test model generation with sample input."""
        model = create_transformer_model(self.model_config)
        model = model.to(self.device)
        model.eval()
        
        # Sample question
        question = "If John has 5 apples and gives 2 to his friend, how many apples does he have left?"
        input_text = f"Question: {question}\nLet's solve this step by step:\n"
        
        inputs = self.tokenizer(
            input_text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=200,
                num_beams=2,
                temperature=0.7,
                pad_token_id=self.tokenizer.pad_token_id
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        self.assertIsInstance(generated_text, str)
        self.assertGreater(len(generated_text), len(input_text))

if __name__ == "__main__":
    unittest.main()
