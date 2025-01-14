import pandas as pd
import os
import json

def generate_train_parquet(output_path):
    # Example data generation logic
    data = {
        "feature1": [1, 2, 3],
        "feature2": ["A", "B", "C"],
        "label": [0, 1, 0]
    }
    df = pd.DataFrame(data)
    df.to_parquet(output_path)
    print(f"Generated training data at {output_path}")

def generate_val_parquet(output_path):
    # Example data generation logic
    data = {
        "feature1": [4, 5, 6],
        "feature2": ["D", "E", "F"],
        "label": [1, 0, 1]
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
    
    generate_train_parquet(train_output)
    generate_val_parquet(val_output)

if __name__ == "__main__":
    main()