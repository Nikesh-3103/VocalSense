import os
import pandas as pd
from data_loader import load_ravdess, load_savee, load_tess, load_crema
from utils import plot_distribution

# Load datasets
df_ravdess = load_ravdess(r"C:\Project\emotion recognition with audio\data\Ravdess")
df_savee = load_savee("data/Savee")
df_tess = load_tess("data/Tess")
df_crema = load_crema(r"C:\Project\emotion recognition with audio\data\Crema")

# Combine them
df = pd.concat([df_ravdess, df_savee, df_tess, df_crema], ignore_index=True)

# Optionally plot distribution
plot_distribution(df)

# Normalize paths by stripping dataset folder name (just keeping filename or relative path)
df['path'] = df['path'].apply(lambda p: os.path.basename(p))  # Just the filename

# Save to CSV
df.to_csv("audio_file_emotions.csv", index=False)
print("✅ Saved CSV with cleaned 'path' and 'emotion'")
