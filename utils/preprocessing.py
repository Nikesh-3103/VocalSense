import librosa
import numpy as np

def extract_features(file_path, mfcc=True, chroma=True, mel=True):
    y, sr = librosa.load(file_path, sr=None)
    features = []

    if mfcc:
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
        features.append(np.mean(mfccs.T, axis=0))

    if chroma:
        stft = np.abs(librosa.stft(y))
        chroma_vals = librosa.feature.chroma_stft(S=stft, sr=sr)
        features.append(np.mean(chroma_vals.T, axis=0))

    if mel:
        mel_vals = librosa.feature.melspectrogram(y, sr=sr)
        features.append(np.mean(mel_vals.T, axis=0))

    return np.hstack(features)
