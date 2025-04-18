import os
import pandas as pd

def load_crema(dataset_path):
    emotion_map = {
        'ANG': 'angry', 'DIS': 'disgust', 'FEA': 'fearful', 'HAP': 'happy',
        'NEU': 'neutral', 'SAD': 'sad'
    }

    data = []
    for file in os.listdir(dataset_path):
        if file.endswith('.wav'):
            parts = file.split('_')
            if len(parts) >= 3:
                emotion_code = parts[2]
                emotion = emotion_map.get(emotion_code, None)
                if emotion:
                    data.append({
                        "path": os.path.join(dataset_path, file),
                        "emotion": emotion,
                        "source": "CREMA-D"
                    })
                else:
                    print(f"Unknown emotion code in file: {file}")
            else:
                print(f"Skipping malformed file: {file}")
    
    return pd.DataFrame(data)
