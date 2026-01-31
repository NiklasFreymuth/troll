import glob
import os

import pandas as pd

# Directory containing eurus datasets
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/eurus")
OUTPUT_SUFFIX = "_math_only.parquet"

# Find all Parquet files in the eurus data directory
parquet_files = [
    os.path.join(DATA_DIR, "train.parquet"),
    os.path.join(DATA_DIR, "validation.parquet"),
]

for file_path in parquet_files:
    df = pd.read_parquet(file_path)
    if "ability" not in df.columns:
        print(f"Skipping {file_path}: no 'ability' column.")
        continue
    math_df = df[df["ability"] == "math"]
    if math_df.empty:
        print(f"No math examples found in {file_path}.")
        continue
    output_path = file_path.replace(".parquet", OUTPUT_SUFFIX)
    math_df.to_parquet(output_path, index=False)
    print(f"Filtered file saved as {output_path}")
