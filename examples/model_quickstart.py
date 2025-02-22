"""
Quickstart example for creating and using different Vishwamai models
"""
import torch
from vishwamai.model import (
    create_causal_lm,
    create_masked_lm,
    create_seq2seq_lm,
    create_classifier,
    create_embedder,
    ModelSize,
    PrecisionMode
)
from vishwamai.utils.t4_utils import enable_t4_optimizations, get_device_capabilities

def main():
    # Enable T4 optimizations
    enable_t4_optimizations()
    
    # Get device capabilities
    capabilities = get_device_capabilities()
    print("Device capabilities:", capabilities)
    
    # Example 1: Create a small causal language model with mixed precision
    print("\nCreating causal language model...")
    causal_lm = create_causal_lm(
        size=ModelSize.SMALL,
        precision=PrecisionMode.FP16,
        flash_attention=capabilities["flash_attention"]
    )
    print(f"Causal LM parameters: {sum(p.numel() for p in causal_lm.parameters()):,}")
    
    # Example 2: Create a base-sized masked language model
    print("\nCreating masked language model...")
    masked_lm = create_masked_lm(
        size=ModelSize.BASE,
        precision=PrecisionMode.FP16 if capabilities["amp"] else PrecisionMode.FP32
    )
    print(f"Masked LM parameters: {sum(p.numel() for p in masked_lm.parameters()):,}")
    
    # Example 3: Create a tiny sequence-to-sequence model
    print("\nCreating sequence-to-sequence model...")
    seq2seq = create_seq2seq_lm(
        size=ModelSize.TINY,
        precision=PrecisionMode.BF16 if capabilities["bfloat16"] else PrecisionMode.FP16
    )
    print(f"Seq2Seq parameters: {sum(p.numel() for p in seq2seq.parameters()):,}")
    
    # Example 4: Create a classifier with custom configuration
    print("\nCreating classifier model...")
    classifier = create_classifier(
        size=ModelSize.SMALL,
        precision=PrecisionMode.FP32,
        num_classes=10,
        dropout_rate=0.2
    )
    print(f"Classifier parameters: {sum(p.numel() for p in classifier.parameters()):,}")
    
    # Example 5: Create an embedder model
    print("\nCreating embedder model...")
    embedder = create_embedder(
        size=ModelSize.TINY,
        precision=PrecisionMode.FP16,
        pooling_type="mean"
    )
    print(f"Embedder parameters: {sum(p.numel() for p in embedder.parameters()):,}")
    
    # Example usage with dummy input
    print("\nTesting models with dummy input...")
    batch_size, seq_length = 2, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length)
    
    # Test causal LM
    with torch.no_grad():
        outputs = causal_lm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        print("\nCausal LM output shape:", outputs["hidden_states"].shape)
    
    # Test masked LM
    with torch.no_grad():
        outputs = masked_lm(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        print("Masked LM output shape:", outputs["hidden_states"].shape)
    
    # Test seq2seq
    decoder_input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    with torch.no_grad():
        outputs = seq2seq(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        print("Seq2Seq output shape:", outputs["hidden_states"].shape)
    
    # Test classifier
    with torch.no_grad():
        outputs = classifier(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        print("Classifier logits shape:", outputs["logits"].shape)
    
    # Test embedder
    with torch.no_grad():
        outputs = embedder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        print("Embedder output shape:", outputs["embeddings"].shape)

if __name__ == "__main__":
    main()
