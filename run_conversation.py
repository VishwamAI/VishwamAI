import torch
from vishwamai.model import VishwamaiConfig, VishwamaiModel  # Updated import
from vishwamai.conceptual_tokenizer import ConceptualTokenizer, ConceptualTokenizerConfig
from transformers import PreTrainedTokenizer
import readline  # Optional, for command-line history and editing

def load_model(model_path: str) -> VishwamaiModel:
    """
    Load the Vishwamai model from the specified path.

    Args:
        model_path (str): Path to the pretrained model directory.

    Returns:
        VishwamaiModel: The loaded Vishwamai model.
    """
    model = VishwamaiModel.from_pretrained(model_path)  # Updated to use from_pretrained
    model.eval()
    return model

def load_tokenizer() -> ConceptualTokenizer:
    tokenizer_config = ConceptualTokenizerConfig()
    tokenizer = ConceptualTokenizer(tokenizer_config)
    tokenizer.load_tokenizer()
    return tokenizer

def generate_response(model: VishwamaiModel, tokenizer: ConceptualTokenizer, prompt: str, max_length: int = 50) -> str:
    input_ids = tokenizer.encode(prompt, add_special_tokens=True)
    input_tensor = torch.tensor([input_ids])

    # Generate response using the model
    with torch.no_grad():
        output_ids = model.generate(input_tensor, max_length=max_length)
    
    # Decode the generated tokens to text
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return response

def main():
    # Load the model and tokenizer
    model_path = "/home/kasinadhsarma/VishwamAI/models/vishwamai_model"  # Ensure this path contains config.json and pytorch_model.bin
    model = load_model(model_path)
    tokenizer = load_tokenizer()

    print("VishwamAI Conversation Mode. Type 'exit' to quit.")
    while True:
        try:
            prompt = input("You: ")
            if prompt.lower() in {"exit", "quit"}:
                print("Exiting VishwamAI. Goodbye!")
                break
            response = generate_response(model, tokenizer, prompt)
            print(f"VishwamAI: {response}")
        except KeyboardInterrupt:
            print("\nExiting VishwamAI. Goodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()