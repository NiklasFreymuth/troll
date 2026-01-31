import math
import os

import pandas as pd

# Directory containing eurus datasets
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/eurus")
OUTPUT_SUFFIX = "_code_only"
CHUNK_SIZE = 4096

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

    # drop all the non-code related rows
    mask = df["ability"] == "code"
    to_drop = df.index[~mask]
    df.drop(to_drop, inplace=True)
    df.reset_index(drop=True, inplace=True)

    if df.empty:
        print(f"No code examples found in {file_path}.")
        continue

    # chunk into smaller parquets, since verl dataset load somehow can't handle this one
    num_filtered_rows = len(df)
    n_parts = math.ceil(num_filtered_rows / CHUNK_SIZE)
    for i in range(n_parts):
        start = i * CHUNK_SIZE
        end = start + CHUNK_SIZE
        chunk = df.iloc[start:end]
        filename = file_path.replace(".parquet", f"{OUTPUT_SUFFIX}_{i}.parquet")
        chunk.to_parquet(filename, index=False)

        print(f"Filtered file chunk saved as {filename}")
