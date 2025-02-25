import unittest
import sys
import os
import time  # Import the time module

# Add the parent directory to the path so we can import from vishwamai
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

class TestImports(unittest.TestCase):
    test_counter = 0  # Class level variable

    @classmethod
    def setUpClass(cls):
        """Set up for the tests - introduce a delay."""
        print("Setting up test environment for all tests...")
        time.sleep(1)  # Pause for 1 second to simulate setup time
        print("Test environment ready for all tests.")

    @classmethod
    def tearDownClass(cls):
        print("Cleaning up the test environment for all tests...")
        time.sleep(0.5)
        print("Test environment cleaned up for all tests.\n")

    def setUp(self):
        """Set up for each test - introduce a delay."""
        print(f"Setting up test environment for a single test, current test counter = {self.test_counter}...")
        time.sleep(0.2)  # Pause for 0.2 second to simulate setup time
        print(f"Test environment ready for a single test, current test counter = {self.test_counter}.")

    def tearDown(self):
        """Clean up after each test."""
        print(f"Cleaning up test environment for a single test, current test counter = {self.test_counter}...")
        time.sleep(0.2)  # Pause for 0.2 seconds to simulate cleanup time
        print(f"Test environment cleaned up for a single test, current test counter = {self.test_counter}.\n")
        TestImports.test_counter += 1

    def test_model_imports(self):
        """Test imports from model module"""
        print(f"Running test_model_imports, test counter = {self.test_counter} ...")
        time.sleep(0.5)  # added time
        from vishwamai.model import VishwamAIModel, ModelConfig
        self.assertTrue(True)  # If we got here without error, it's a pass

    def test_tokenizer_imports(self):
        """Test imports from tokenizer module"""
        print(f"Running test_tokenizer_imports, test counter = {self.test_counter} ...")
        time.sleep(0.5)  # added time
        from vishwamai.tokenizer import VishwamAITokenizer
        self.assertTrue(True)

    def test_transformer_imports(self):
        """Test imports from transformer module"""
        print(f"Running test_transformer_imports, test counter = {self.test_counter} ...")
        time.sleep(0.5)  # added time
        from vishwamai.transformer import VishwamAIModel, VisionTransformer10B
        self.assertTrue(True)

    def test_tot_imports(self):
        """Test imports from tot module"""
        print(f"Running test_tot_imports, test counter = {self.test_counter} ...")
        time.sleep(0.5)  # added time
        try:
            from vishwamai.tot import TreeOfThoughts, Thought, SearchState
            self.assertTrue(True)
        except ImportError as e:
            print(f"TOT Import error detail: {str(e)}")
            self.fail(f"Failed to import from tot module: {str(e)}")

    def test_distillation_imports(self):
        """Test imports from distillation module"""
        print(f"Running test_distillation_imports, test counter = {self.test_counter} ...")
        time.sleep(0.5)  # added time
        try:
            from vishwamai.distillation import VishwamaiGuruKnowledge, VishwamaiShaalaTrainer
            self.assertTrue(True)
        except ImportError as e:
            print(f"Distillation Import error detail: {str(e)}")
            self.fail(f"Failed to import from distillation module: {str(e)}")

    def test_data_utils_imports(self):
        """Test imports from data_utils module"""
        print(f"Running test_data_utils_imports, test counter = {self.test_counter} ...")
        time.sleep(0.5)  # added time
        try:
            from vishwamai.data_utils import create_train_dataloader, create_val_dataloader
            self.assertTrue(True)
        except ImportError as e:
            print(f"Data utils Import error detail: {str(e)}")
            self.fail(f"Failed to import from data_utils module: {str(e)}")

    def test_integration_imports(self):
        """Test imports from integration module"""
        print(f"Running test_integration_imports, test counter = {self.test_counter} ...")
        time.sleep(0.5)  # added time
        try:
            from vishwamai.integration import ToTIntegrationLayer, MixtureDensityNetwork
            self.assertTrue(True)
        except ImportError as e:
            print(f"Integration Import error detail: {str(e)}")
            self.fail(f"Failed to import from integration module: {str(e)}")

    def test_loop(self):
        """Test a while loop"""
        print(f"Running a while loop test, test counter = {self.test_counter}...")
        test_counter = 0
        while test_counter < 3:  # Run the loop three times
            print(f"Loop iteration: {test_counter}")
            time.sleep(1)  # Pause for 1 second
            test_counter += 1
        print(f"While loop test completed, test counter = {self.test_counter}.")
        self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
