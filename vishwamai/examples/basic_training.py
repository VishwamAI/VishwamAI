import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from vishwamai.trainer_unified import UnifiedTrainer, UnifiedTrainerConfig
from vishwamai.model_factory import AdvancedModelConfig
from vishwamai.curriculum import CurriculumConfig
from vishwamai.emergent_behavior import EmergentConfig
from vishwamai.integrated_information import IntegrationConfig
from vishwamai.ethical_framework import EthicalConfig
from vishwamai.hardware_adapter import HardwareConfig
from vishwamai.open_ended_learning import OpenEndedConfig

def create_sample_dataset():
    """Create a simple dataset for demonstration."""
    # Sample text for training
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is a subset of artificial intelligence.",
        "Neural networks can learn complex patterns in data.",
        "Ethics in AI is becoming increasingly important.",
        "Consciousness remains a fascinating area of study."
    ]
    
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    encodings = tokenizer(
        texts,
        padding=True,
        truncation=True,
        max_length=64,
        return_tensors='pt'
    )
    
    # Create labels for language modeling
    labels = encodings.input_ids.clone()
    
    # Create dataset
    dataset = torch.utils.data.TensorDataset(
        encodings.input_ids,
        encodings.attention_mask,
        labels
    )
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=2,
        shuffle=True
    )

def main():
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load base model
    model = AutoModelForCausalLM.from_pretrained('gpt2-small')
    model = model.to(device)
    
    # Configure components
    model_config = AdvancedModelConfig(
        hidden_dim=768,  # GPT-2 small hidden size
        emergent_config=EmergentConfig(
            min_novelty_threshold=0.3,
            exploration_rate=0.1
        ),
        integration_config=IntegrationConfig(
            phi_threshold=0.5,
            integration_window=10
        ),
        ethical_config=EthicalConfig(
            min_ethical_score=0.7,
            transparency_level=0.8
        ),
        hardware_config=HardwareConfig(
            precision="float32",
            enable_tensor_cores=True
        ),
        open_ended_config=OpenEndedConfig(
            min_novelty_threshold=0.3,
            max_complexity=5.0
        )
    )
    
    # Create trainer configuration
    trainer_config = UnifiedTrainerConfig(
        model_config=model_config,
        curriculum_config=CurriculumConfig(
            min_sequence_length=32,
            max_sequence_length=64
        ),
        mixed_precision=True,
        gradient_checkpointing=True,
        gradient_accumulation_steps=1,
        logging_steps=1,
        eval_steps=5,
        save_steps=10
    )
    
    # Initialize trainer
    trainer = UnifiedTrainer(
        model=model,
        config=trainer_config,
        device=device
    )
    
    # Create dataloaders
    train_dataloader = create_sample_dataset()
    eval_dataloader = create_sample_dataset()
    
    # Training loop
    num_epochs = 3
    print("Starting training...")
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Prepare batch
            batch = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'labels': batch[2]
            }
            
            # Training step
            metrics = trainer.train_step(batch)
            
            # Log metrics
            if trainer.global_step % trainer.config.logging_steps == 0:
                print(f"\nStep {trainer.global_step}")
                print(f"Loss: {metrics['loss']:.4f}")
                print(f"Emergent Score: {metrics.get('emergent_loss', 0):.4f}")
                print(f"Ethical Score: {metrics.get('ethical_score', 0):.4f}")
                
            # Evaluation
            if trainer.global_step % trainer.config.eval_steps == 0:
                eval_metrics = trainer.evaluate(eval_dataloader)
                print(f"\nEvaluation metrics:")
                print(f"Eval Loss: {eval_metrics['eval_loss']:.4f}")
                
            # Save checkpoint
            if trainer.global_step % trainer.config.save_steps == 0:
                trainer.save_checkpoint(f'checkpoint_step_{trainer.global_step}.pt')
                
        # End of epoch
        print(f"\nEpoch {epoch + 1} completed")
        print("Component metrics:")
        component_stats = trainer.components['emergent'].get_learning_trajectory()
        print(f"Learning complexity: {component_stats['complexity_trend'][-1]:.4f}")
        print(f"Novelty score: {component_stats['novelty_trend'][-1]:.4f}")
        
    print("\nTraining completed!")
    
    # Final save
    trainer.save_checkpoint('final_model.pt')
    print("Model saved to 'final_model.pt'")

if __name__ == '__main__':
    main()
