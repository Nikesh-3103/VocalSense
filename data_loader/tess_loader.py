import os
import pandas as pd

def load_tess(dataset_path):
    data = []
    for folder in os.listdir(dataset_path):
        emotion = folder.split('_')[-1].lower()
        folder_path = os.path.join(dataset_path, folder)
        for file in os.listdir(folder_path):
            if file.endswith('.wav'):
                data.append({"path": os.path.join(folder_path, file), "emotion": emotion, "source": "TESS"})
    
    return pd.DataFrame(data)
