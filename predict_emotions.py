import sys
import torch
import librosa
import numpy as np
import matplotlib.pyplot as plt
from transformers import Wav2Vec2Processor, Wav2Vec2Model
from keras.models import load_model
import keras.backend as K
import tensorflow as tf

# Define a dummy focal loss function (same name, so Keras can match it)
def focal_loss(y_true, y_pred, gamma=2.0, alpha=0.25):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.clip_by_value(y_pred, K.epsilon(), 1.0 - K.epsilon())
    cross_entropy = -y_true * tf.math.log(y_pred)
    weight = alpha * tf.math.pow(1 - y_pred, gamma)
    loss = weight * cross_entropy
    return tf.reduce_mean(tf.reduce_sum(loss, axis=1))

# Update this list to match the exact order used during training
EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'calm', 'sad', 'surprise', 'ps', 'bored', 'excited']

# Load Wav2Vec2 model & processor
processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base")
wav2vec_model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base")

# Load your trained 
model = load_model("final_emotion_model.keras", custom_objects={"focal_loss": focal_loss})

def extract_wav2vec_features(audio_path):
    signal, sr = librosa.load(audio_path, sr=16000)
    inputs = processor(signal, sampling_rate=16000, return_tensors="pt", padding=True)
    with torch.no_grad():
        features = wav2vec_model(**inputs).last_hidden_state
    return features.squeeze(0).numpy()  # Shape: (time, 768)

def predict_emotion(audio_path):
    features = extract_wav2vec_features(audio_path)
    
    # Optional: apply mean pooling across time
    mean_pooled = np.mean(features, axis=0)

    # Model expects (batch_size, time, features), so reshape accordingly
    model_input = np.expand_dims(mean_pooled, axis=0)

    predictions = model.predict(model_input)
    
    # Plot emotion probabilities
    plt.figure(figsize=(10, 5))
    plt.bar(EMOTIONS, predictions[0], color='skyblue')
    plt.ylabel("Probability")
    plt.title("Emotion Prediction")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()

    predicted_label = EMOTIONS[np.argmax(predictions)]
    print(f"\nPredicted Emotion: {predicted_label}")
    return predicted_label

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python predict_emotions.py <audio_path>")
        sys.exit(1)

    audio_path = sys.argv[1]
    predict_emotion(audio_path)
