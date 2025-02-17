import pandas as pd

# Load the Parquet file
parquet_file = "/home/kasinadhsarma/VishwamAI/gsm8k/train-00000-of-00001.parquet"
df = pd.read_parquet(parquet_file, engine="pyarrow")

# Convert to CSV
csv_file = "/home/kasinadhsarma/VishwamAI/gsm8k/train.csv"
df.to_csv(csv_file, index=False)

print(f"Converted CSV file saved at: {csv_file}")
