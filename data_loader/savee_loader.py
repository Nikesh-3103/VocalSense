import os
import pandas as pd

def load_savee(dataset_path):
    emotion_map = {
        'a': 'angry', 'd': 'disgust', 'f': 'fearful', 'h': 'happy',
        'n': 'neutral', 'sa': 'sad', 'su': 'surprised'
    }

    data = []
    for file in os.listdir(dataset_path):
        if file.endswith('.wav'):
            code = file[:2] if file[:2] in emotion_map else file[0]
            emotion = emotion_map.get(code, 'unknown')
            data.append({"path": os.path.join(dataset_path, file), "emotion": emotion, "source": "SAVEE"})
    
    return pd.DataFrame(data)
