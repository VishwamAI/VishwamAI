import pandas as pd
import os
import json
import numpy as np

def generate_train_parquet(output_path, vocab_size):
    # Example data generation logic with token indices within vocab_size
    data = {
        "feature1": [1, 2, 3],
        "feature2": ["A", "B", "C"],
        "label": [0, 1, 0],
        "tokens": [np.random.randint(0, vocab_size, size=32).tolist() for _ in range(3)]
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"Generated training data at {output_path}")

def generate_val_parquet(output_path, vocab_size):
    # Example data generation logic with token indices within vocab_size
    data = {
        "feature1": [4, 5, 6],
        "feature2": ["D", "E", "F"],
        "label": [1, 0, 1],
        "tokens": [np.random.randint(0, vocab_size, size=32).tolist() for _ in range(3)]
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"Generated validation data at {output_path}")

def main():
    config_path = os.path.join(os.path.dirname(__file__), "../vishwamai/config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    train_output = config.get("train_data")
    val_output = config.get("val_data")
    vocab_size = config['model_config'].get('vocab_size', 32000)
    
    generate_train_parquet(train_output, vocab_size)
    generate_val_parquet(val_output, vocab_size)

if __name__ == "__main__":
    main()