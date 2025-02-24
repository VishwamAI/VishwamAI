import unittest

class TestImports(unittest.TestCase):
    def test_model_imports(self):
        from vishwamai.model import VishwamAIModel, ModelConfig
        self.assertIsNotNone(VishwamAIModel)
        self.assertIsNotNone(ModelConfig)

    def test_tokenizer_imports(self):
        from vishwamai.tokenizer import VishwamAITokenizer
        self.assertIsNotNone(VishwamAITokenizer)

    def test_distillation_imports(self):
        from vishwamai.distillation import VishwamaiGuruKnowledge, VishwamaiShaalaTrainer
        self.assertIsNotNone(VishwamaiGuruKnowledge)
        self.assertIsNotNone(VishwamaiShaalaTrainer)

    def test_data_utils_imports(self):
        from vishwamai.data_utils import create_train_dataloader, create_val_dataloader
        self.assertIsNotNone(create_train_dataloader)
        self.assertIsNotNone(create_val_dataloader)

if __name__ == '__main__':
    unittest.main()
