import jax
import jax.numpy as jnp
from vishwamai.model import ModelConfig, VishwamAIModel, create_integrated_model
from vishwamai.pipeline import pipeline

def demo_model_usage():
    # Create model configuration
    config = ModelConfig(
        dim=2048,
        depth=32,
        heads=32,
        vocab_size=50304,
        max_seq_len=8192,
        dropout_rate=0.1,
        expert_count=8,
        expert_capacity=4
    )
    
    # Create integrated model with all components
    model_dict = create_integrated_model(config)
    model = model_dict['model']
    
    # Initialize model parameters with a sample input
    rng = jax.random.PRNGKey(0)
    sample_input = jnp.ones((1, 10), dtype=jnp.int32)  # Batch size 1, sequence length 10
    params = model.init(rng, sample_input)
    
    # Example text generation with error correction
    input_sequence = jnp.array([[1, 2, 3]], dtype=jnp.int32)  # Replace with actual token IDs
    generated_tokens = model.apply(params, input_sequence)
    
    return generated_tokens

def main():
    # Initialize pipeline with Phi-4
    text_generator = pipeline(
        "text-generation",
        model="microsoft/phi-4",
        model_kwargs={"torch_dtype": "auto"},
        device_map="auto",
    )

    # Example chat messages
    messages = [
        {
            "role": "system",
            "content": "You are a medieval knight and must provide explanations to modern people."
        },
        {
            "role": "user",
            "content": "How should I explain the Internet?"
        }
    ]

    # Generate response
    outputs = text_generator(messages, max_new_tokens=128)
    print(outputs[0]["generated_text"])

if __name__ == "__main__":
    main()
