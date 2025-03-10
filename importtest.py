"""
VishwamAI Transformer Implementation - Import Test
"""

import unittest
import jax
import jax.numpy as jnp
from vishwamai import (
    ChainOfThoughtPrompting,
    TreeOfThoughts,
    ThoughtNode,
    evaluate_tot_solution,
    compute_distillation_loss,
    create_student_model,
    initialize_from_teacher,
    create_vishwamai_transformer,
    create_train_state,
    fp8_cast_transpose,
    fp8_gemm_optimized,
    block_tpu_matmul,
    act_quant,
    multi_head_attention_kernel,
    flash_attention,
    rope_embedding,
    apply_rotary_pos_emb,
    weight_dequant,
    batch_norm_tpu,
    fp8_gemm,
    MLABlock,
    MoELayer,
    VishwamAIPipeline,
    VishwamAITrainer,
    create_trainer,
    DuckDBLogger,
)
from vishwamai.transformer import TransformerModel, EnhancedTransformerModel
from vishwamai.tokenizer import VishwamAITokenizer
from vishwamai.model import VishwamAI

class TestImports(unittest.TestCase):
    """
    Test cases to ensure all modules and functions are importable.
    """

    def test_core_components(self):
        # Test ChainOfThoughtPrompting
        self.assertIsNotNone(ChainOfThoughtPrompting)

        # Test TreeOfThoughts
        self.assertIsNotNone(TreeOfThoughts)

        # Test ThoughtNode
        self.assertIsNotNone(ThoughtNode)
        
        # Test evaluate_tot_solution
        self.assertIsNotNone(evaluate_tot_solution)

        # Test compute_distillation_loss
        self.assertIsNotNone(compute_distillation_loss)

        # Test create_student_model
        self.assertIsNotNone(create_student_model)

        # Test initialize_from_teacher
        self.assertIsNotNone(initialize_from_teacher)

        # Test create_vishwamai_transformer
        self.assertIsNotNone(create_vishwamai_transformer)

        # Test create_train_state
        self.assertIsNotNone(create_train_state)
        
        
    def test_tpu_optimized_kernels(self):
        # Test fp8_cast_transpose
        self.assertIsNotNone(fp8_cast_transpose)

        # Test fp8_gemm_optimized
        self.assertIsNotNone(fp8_gemm_optimized)

        # Test block_tpu_matmul
        self.assertIsNotNone(block_tpu_matmul)

        # Test act_quant
        self.assertIsNotNone(act_quant)

        # Test multi_head_attention_kernel
        self.assertIsNotNone(multi_head_attention_kernel)

        # Test flash_attention
        self.assertIsNotNone(flash_attention)

        # Test rope_embedding
        self.assertIsNotNone(rope_embedding)

        # Test apply_rotary_pos_emb
        self.assertIsNotNone(apply_rotary_pos_emb)

        # Test weight_dequant
        self.assertIsNotNone(weight_dequant)

        # Test batch_norm_tpu
        self.assertIsNotNone(batch_norm_tpu)
        
        #Test fp8_gemm
        self.assertIsNotNone(fp8_gemm)

    def test_advanced_layers(self):
        # Test MLABlock
        self.assertIsNotNone(MLABlock)

        # Test MoELayer
        self.assertIsNotNone(MoELayer)

    def test_pipeline_and_training(self):
        # Test VishwamAIPipeline
        self.assertIsNotNone(VishwamAIPipeline)

        # Test VishwamAITrainer
        self.assertIsNotNone(VishwamAITrainer)

        # Test create_trainer
        self.assertIsNotNone(create_trainer)

    def test_logging(self):
        # Test DuckDBLogger
        self.assertIsNotNone(DuckDBLogger)
    
    def test_transformer_models(self):
        self.assertIsNotNone(TransformerModel)
        self.assertIsNotNone(EnhancedTransformerModel)
    
    def test_tokenizer(self):
        self.assertIsNotNone(VishwamAITokenizer)

    def test_vishwamai_model(self):
        self.assertIsNotNone(VishwamAI)

    def test_example_usage(self):
        # Check if we can create an instance of DuckDBLogger
        logger = DuckDBLogger(db_path=":memory:")
        self.assertIsNotNone(logger)
        logger.close()

        # Check if we can create an instance of EnhancedTransformerModel with TPU v2 optimized config
        config = {
            'vocab_size': 1000,
            'num_layers': 2,
            'num_heads': 4,
            'head_dim': 32,  # Reduced for TPU v2 memory efficiency
            'hidden_dim': 128,  # Adjusted for TPU v2
            'mlp_dim': 256,  # Adjusted for TPU v2
            'max_seq_len': 64,  # Reduced for testing
            'use_enhanced': True,
            'use_rotary': True,
            'use_flash_attn': True,
            'use_rms_norm': False,  # Using LayerNorm for TPU v2 stability
            'dropout_rate': 0.1,
            'dtype': jnp.float32  # Using float32 for TPU v2
        }

        # Create model and validate
        model = create_vishwamai_transformer(config)
        self.assertIsNotNone(model)

        # Initialize training state
        rng = jax.random.PRNGKey(0)
        learning_rate_schedule = lambda step: 1e-3
        state = create_train_state(rng, config, learning_rate_schedule)
        self.assertIsNotNone(state)

        # Test forward pass with proper shape
        batch_size, seq_len = 1, 16
        input_data = jnp.ones((batch_size, seq_len), dtype=jnp.int32)
        output = model.apply({'params': state.params}, input_data, deterministic=True)
        self.assertEqual(output.shape, (batch_size, seq_len, config['vocab_size']))

        # Test tokenizer
        tokenizer = VishwamAITokenizer(model_path="test.model")
        self.assertIsNotNone(tokenizer)

        # Test VishwamAI model
        vishwamai_model = VishwamAI(vocab_size=1000)
        self.assertIsNotNone(vishwamai_model)
        
        
if __name__ == '__main__':
    unittest.main()

