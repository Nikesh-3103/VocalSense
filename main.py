import os
import pandas as pd
import numpy as np
from data_loader import load_ravdess, load_savee, load_tess, load_crema
from utils import extract_features, plot_distribution
from tqdm import tqdm

# -----------------------
# Load Datasets
# -----------------------
df_ravdess = load_ravdess(r"C:\Project\emotion recognition with audio\data\Ravdess")
df_savee = load_savee("data/Savee")
df_tess = load_tess("data/Tess")
df_crema = load_crema(r"C:\Project\emotion recognition with audio\data\Crema")

# Combine them
df = pd.concat([df_ravdess, df_savee, df_tess, df_crema], ignore_index=True)

# Plot class distribution
plot_distribution(df)

# -----------------------
# Feature Extraction
# -----------------------
features = []
labels = []

for i, row in tqdm(df.iterrows(), total=len(df)):
    try:
        feat = extract_features(row['path'])  # Assume 1D feature array like MFCCs
        features.append(feat)
        labels.append(row['emotion'])
    except Exception as e:
        print(f"Error processing {row['path']}: {e}")

# -----------------------
# Convert to DataFrame
# -----------------------
features_df = pd.DataFrame(features)
features_df['emotion'] = labels

# Save to CSV
features_df.to_csv("features_labeled_dataset.csv", index=False)
print("Dataset saved as features_labeled_dataset.csv")
