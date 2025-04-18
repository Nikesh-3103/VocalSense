import os
import pandas as pd

def load_ravdess(dataset_path):
    emotion_map = {
        '01': 'neutral',
        '02': 'calm',
        '03': 'happy',
        '04': 'sad',
        '05': 'angry',
        '06': 'fearful',
        '07': 'disgust',
        '08': 'surprised'
    }

    data = []
    for root, _, files in os.walk(dataset_path):
        for file in files:
            if file.endswith(".wav"):
                emotion_code = file.split('-')[2]
                emotion = emotion_map.get(emotion_code, 'unknown')
                data.append({"path": os.path.join(root, file), "emotion": emotion, "source": "RAVDESS"})
    
    return pd.DataFrame(data)
