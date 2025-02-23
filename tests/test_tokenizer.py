import unittest
import tempfile
import os
from pathlib import Path
from vishwamai.tokenizer import VishwamAITokenizer

class TestVishwamAITokenizer(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        
        # Create sample text file for training
        self.train_file = os.path.join(self.test_dir, "train.txt")
        with open(self.train_file, "w") as f:
            f.write("This is a test sentence.\nAnother test sentence for tokenization.")
            
        # Initialize tokenizer
        self.tokenizer = VishwamAITokenizer(
            vocab_size=1000,
            model_prefix="test_tokenizer"
        )

    def test_train_and_encode_decode(self):
        # Train tokenizer
        self.tokenizer.train(self.train_file, self.test_dir)
        
        # Test encoding
        test_text = "Hello world!"
        encoded = self.tokenizer.encode(test_text)
        self.assertIsInstance(encoded, list)
        self.assertTrue(all(isinstance(x, int) for x in encoded))
        
        # Test decoding
        decoded = self.tokenizer.decode(encoded)
        self.assertIsInstance(decoded, str)
        
    def test_save_and_load(self):
        # Train and save
        self.tokenizer.train(self.train_file, self.test_dir)
        save_dir = os.path.join(self.test_dir, "saved_tokenizer")
        self.tokenizer.save(save_dir)
        
        # Load and verify
        loaded_tokenizer = VishwamAITokenizer.from_pretrained(save_dir)
        
        test_text = "Hello world!"
        original_encoding = self.tokenizer.encode(test_text)
        loaded_encoding = loaded_tokenizer.encode(test_text)
        
        self.assertEqual(original_encoding, loaded_encoding)

    def test_batch_processing(self):
        self.tokenizer.train(self.train_file, self.test_dir)
        
        # Test batch encoding
        texts = ["Hello world!", "Another test."]
        encoded = self.tokenizer.encode(texts)
        self.assertIsInstance(encoded, list)
        self.assertEqual(len(encoded), 2)
        
        # Test batch decoding
        decoded = self.tokenizer.decode(encoded)
        self.assertIsInstance(decoded, list)
        self.assertEqual(len(decoded), 2)

    def test_special_tokens(self):
        self.tokenizer.train(self.train_file, self.test_dir)
        
        # Test with special tokens
        test_text = "Hello world!"
        encoded_with_special = self.tokenizer.encode(test_text, add_special_tokens=True)
        encoded_without_special = self.tokenizer.encode(test_text, add_special_tokens=False)
        
        self.assertGreater(len(encoded_with_special), len(encoded_without_special))
        self.assertEqual(encoded_with_special[0], self.tokenizer.bos_id)
        self.assertEqual(encoded_with_special[-1], self.tokenizer.eos_id)

    def tearDown(self):
        # Cleanup temporary files
        for path in Path(self.test_dir).glob("*"):
            path.unlink()
        Path(self.test_dir).rmdir()

if __name__ == "__main__":
    unittest.main()
