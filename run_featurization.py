import pandas as pd
from pathlib import Path
from src.features.feature_engineering import generate_features_and_labels

# Ensure the output directory exists
Path("data/features").mkdir(parents=True, exist_ok=True)

# 1. Read the raw logs we just generated
print("Reading raw logs...")
df = pd.read_parquet("data/raw/security_logs.parquet")

# 2. Apply our new "Security-Correct" logic
print("Applying feature engineering & labeling...")
df = generate_features_and_labels(df)

# 3. Save the ready-for-training data
output_path = "data/features/security_features.parquet"
df.to_parquet(output_path, index=False)

print(f"✅ Success! Feature data saved to: {output_path}")
print(f"   Shape: {df.shape}")
print(f"   Attack/Benign Split:\n{df['label'].value_counts()}")