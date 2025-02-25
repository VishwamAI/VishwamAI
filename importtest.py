import unittest
import sys
import os

# Add the parent directory to the path so we can import from vishwamai
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestImports(unittest.TestCase):
    
    def test_model_imports(self):
        """Test imports from model module"""
        from vishwamai.model import VishwamAIModel, ModelConfig
        self.assertTrue(True)  # If we got here without error, it's a pass
    
    def test_tokenizer_imports(self):
        """Test imports from tokenizer module"""
        from vishwamai.tokenizer import VishwamAITokenizer
        self.assertTrue(True)
        
    def test_transformer_imports(self):
        """Test imports from transformer module"""
        from vishwamai.transformer import VishwamAIModel, VisionTransformer10B
        self.assertTrue(True)
        
    def test_tot_imports(self):
        """Test imports from tot module"""
        try:
            from vishwamai.tot import TreeOfThoughts, Thought, SearchState
            self.assertTrue(True)
        except ImportError as e:
            print(f"TOT Import error detail: {str(e)}")
            self.fail(f"Failed to import from tot module: {str(e)}")
    
    def test_distillation_imports(self):
        """Test imports from distillation module"""
        try:
            from vishwamai.distillation import VishwamaiGuruKnowledge, VishwamaiShaalaTrainer
            self.assertTrue(True)
        except ImportError as e:
            print(f"Distillation Import error detail: {str(e)}")
            self.fail(f"Failed to import from distillation module: {str(e)}")
    
    def test_data_utils_imports(self):
        """Test imports from data_utils module"""
        try:
            from vishwamai.data_utils import create_train_dataloader, create_val_dataloader
            self.assertTrue(True)
        except ImportError as e:
            print(f"Data utils Import error detail: {str(e)}")
            self.fail(f"Failed to import from data_utils module: {str(e)}")
            
    def test_integration_imports(self):
        """Test imports from integration module"""
        try:
            from vishwamai.integration import ToTIntegrationLayer, MixtureDensityNetwork
            self.assertTrue(True)
        except ImportError as e:
            print(f"Integration Import error detail: {str(e)}")
            self.fail(f"Failed to import from integration module: {str(e)}")

if __name__ == '__main__':
    unittest.main()
