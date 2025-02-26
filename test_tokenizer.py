import os
import sys
import logging
from vishwamai.tokenizer import VishwamAITokenizer

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_tokenizer():
    """Test the tokenizer functionality with a small sample."""
    try:
        # Create a small test file with more content
        test_file = "tokenizer_test_data.txt"
        with open(test_file, "w") as f:
            f.write("""This is a test sentence for the tokenizer.
123 + 456 = 579
How much is 7 * 8?
The answer is 56.
Let's try some mathematical operations:
1 + 1 = 2
10 - 5 = 5
3 * 4 = 12
20 / 4 = 5
2^3 = 8
Square root of 16 is 4.
Pi is approximately 3.14159.
In mathematics, e is approximately 2.71828.
A simple equation: y = mx + b
For a right triangle: a^2 + b^2 = c^2
The quadratic formula: x = (-b ± √(b^2 - 4ac)) / 2a
""")
        
        output_dir = "tokenizer_test_output"
        os.makedirs(output_dir, exist_ok=True)
        
        # Test basic initialization and training with appropriate vocab size
        logger.info("Training basic tokenizer...")
        tokenizer = VishwamAITokenizer(vocab_size=100)  # Much smaller vocab size for this tiny dataset
        tokenizer.train([test_file], output_dir)
        
        # Test encoding and decoding
        test_text = "Testing math: 42 + 17 = 59"
        encoded = tokenizer.encode(test_text)
        decoded = tokenizer.decode(encoded)
        
        logger.info(f"Test text: {test_text}")
        logger.info(f"Encoded: {encoded}")
        logger.info(f"Decoded: {decoded}")
        
        # Test with special tokens
        logger.info("Training tokenizer with special tokens...")
        special_tokens_tokenizer = VishwamAITokenizer(
            vocab_size=100,  # Keep vocab size small for test
            special_tokens=["<math>", "<test>"]
        )
        special_tokens_tokenizer.train([test_file], output_dir + "_special")
        
        # Test special token handling
        math_text = "Let's do <math>1 + 1 = 2</math>"
        math_encoded = special_tokens_tokenizer.encode(math_text)
        math_decoded = special_tokens_tokenizer.decode(math_encoded)
        
        logger.info(f"Math text: {math_text}")
        logger.info(f"Math encoded: {math_encoded}")
        logger.info(f"Math decoded: {math_decoded}")
        
        # Test saving and loading
        save_dir = output_dir + "_saved"
        tokenizer.save(save_dir)
        loaded_tokenizer = VishwamAITokenizer.from_pretrained(save_dir)
        
        # Verify loaded tokenizer works
        loaded_encoded = loaded_tokenizer.encode(test_text)
        loaded_decoded = loaded_tokenizer.decode(loaded_encoded)
        
        logger.info("Testing loaded tokenizer")
        logger.info(f"Loaded encoded: {loaded_encoded}")
        logger.info(f"Loaded decoded: {loaded_decoded}")
        
        logger.info("Tokenizer tests completed successfully")
        
        # Clean up
        os.remove(test_file)
        return True
    except Exception as e:
        logger.error(f"Tokenizer test failed: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_tokenizer()
    sys.exit(0 if success else 1)
