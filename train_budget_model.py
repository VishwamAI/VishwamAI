"""Train a budget-optimized VishwamAI model on TPU v3-8."""

import jax
import tensorflow as tf
from absl import app, flags
from vishwamai.configs.budget_model_config import BudgetModelConfig
from vishwamai.training.budget_trainer import BudgetModelTrainer
from vishwamai.transformer import EnhancedTransformerModel
from vishwamai.tpu_credit_manager import TPUCreditConfig
from vishwamai.device_mesh import TPUMeshContext

FLAGS = flags.FLAGS
flags.DEFINE_string("data_path", None, "Path to training data")
flags.DEFINE_integer("max_steps", None, "Override default training steps")
flags.DEFINE_float("learning_rate", None, "Override default learning rate")
flags.DEFINE_float("max_credits", 112390.0, "Maximum TPU credits available")

def create_dataset(data_path: str, config: BudgetModelConfig):
    """Create training dataset."""
    def parse_function(example):
        features = {
            "input_ids": tf.io.FixedLenFeature([config.model_config["max_position_embeddings"]], tf.int64)
        }
        parsed = tf.io.parse_single_example(example, features)
        return {
            "input_ids": tf.cast(parsed["input_ids"], tf.int32)
        }
    
    # Create dataset from files
    files = tf.data.Dataset.list_files(f"{data_path}/train-*.tfrecord")
    dataset = tf.data.TFRecordDataset(files)
    
    # Parse and batch
    return (
        dataset.map(parse_function)
        .shuffle(10000)
        .batch(config.training_config["batch_size"], drop_remainder=True)
        .prefetch(tf.data.AUTOTUNE)
        .as_numpy_iterator()
    )

def main(_):
    # Initialize configurations
    config = BudgetModelConfig()
    credit_config = TPUCreditConfig(max_credits=FLAGS.max_credits)
    
    # Override configs if specified
    if FLAGS.max_steps:
        config.training_config["max_steps"] = FLAGS.max_steps
    if FLAGS.learning_rate:
        config.training_config["learning_rate"] = FLAGS.learning_rate
    
    # Create TPU mesh context
    mesh_context = TPUMeshContext(config.tpu_config)
    
    # Create model
    model = EnhancedTransformerModel(config=config.model_config)
    
    # Initialize trainer
    trainer = BudgetModelTrainer(config, credit_config)
    
    # Create dataset
    train_ds = create_dataset(FLAGS.data_path, config)
    
    print("Starting budget model training...")
    print(f"Model size: {config.get_parameter_count():,} parameters")
    print(f"Estimated memory usage: {config.get_estimated_memory():.2f} GB")
    print(f"Theoretical speedup: {config.get_theoretical_speedup():.1f}x")
    
    # Train model
    with mesh_context:
        final_state = trainer.train(
            train_ds=train_ds,
            model=model,
            learning_rate=config.training_config["learning_rate"],
            num_steps=config.training_config["max_steps"]
        )
    
    print("Training complete!")
    metrics = trainer.credit_manager.get_credit_metrics()
    print(f"Total credits used: {metrics['credits_used']:.2f}")
    print(f"Average compute utilization: {metrics['avg_utilization']:.1%}")

if __name__ == "__main__":
    app.run(main)