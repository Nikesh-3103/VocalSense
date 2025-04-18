import os
import numpy as np
import pandas as pd
import librosa
from tqdm import tqdm
import joblib

from transformers import Wav2Vec2Processor, Wav2Vec2Model
import torch
from sklearn.preprocessing import LabelEncoder

# ------------------------------
# CONFIGURATION
# ------------------------------
INPUT_CSV = r"C:\Project\emotion recognition with audio\audio_paths_labeled.csv"  # CSV must contain "path", "emotion", "source"
OUTPUT_CSV = "features_wav2vec2.csv"
LABEL_ENCODER_PATH = "label_encoder.pkl"
SAMPLE_RATE = 16000

# ------------------------------
# LOAD Wav2Vec2 MODEL & PROCESSOR
# ------------------------------
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ------------------------------
# FEATURE EXTRACTION FUNCTION
# ------------------------------
def extract_wav2vec2_features(file_path):
    audio, sr = librosa.load(file_path, sr=SAMPLE_RATE)
    if len(audio) == 0:
        raise ValueError("Empty audio file")

    inputs = processor(audio, return_tensors="pt", sampling_rate=SAMPLE_RATE, padding=True)
    with torch.no_grad():
        outputs = model(inputs.input_values.to(device)).last_hidden_state
        features = torch.mean(outputs, dim=1).squeeze().cpu().numpy()

    return features

# ------------------------------
# READ INPUT CSV
# ------------------------------
df = pd.read_csv(INPUT_CSV)
df.columns = df.columns.str.strip()

assert {"path", "emotion", "source"}.issubset(df.columns), "CSV must have columns: path, emotion, source"

features_list = []
emotions = []
dataset_names = []

for _, row in tqdm(df.iterrows(), total=len(df)):
    file_path = row["path"]
    emotion = row["emotion"]
    dataset_name = row["source"]

    try:
        features = extract_wav2vec2_features(file_path)
        features_list.append(features)
        emotions.append(emotion)
        dataset_names.append(dataset_name)
    except Exception as e:
        print(f"⚠️ Skipped {file_path}: {e}")

# ------------------------------
# LABEL ENCODING & SAVE
# ------------------------------
le = LabelEncoder()
labels_encoded = le.fit_transform(emotions)
joblib.dump(le, LABEL_ENCODER_PATH)

features_df = pd.DataFrame(features_list)
features_df["emotion"] = labels_encoded
features_df["dataset_name"] = dataset_names
features_df.to_csv(OUTPUT_CSV, index=False)

print(f"\n✅ Feature extraction complete. Saved to: {OUTPUT_CSV}")
print(f"🔢 Encoded emotions: {list(le.classes_)}")
