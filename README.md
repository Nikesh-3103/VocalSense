# Emotion Recognition with Audio

This project implements an **Emotion Recognition System** using audio data. The model is built from scratch and trained on multiple datasets to classify emotions based on audio features. The system leverages deep learning techniques, including convolutional layers, recurrent layers, and attention mechanisms, to achieve high accuracy.

## Table of Contents
- [Project Overview](#project-overview)
- [Datasets](#datasets)
- [Model Architecture](#model-architecture)
- [Usage](#usage)
- [Folder Structure](#folder-structure)
- [Acknowledgments](#acknowledgments)

---

## Project Overview

The goal of this project is to classify emotions from audio recordings. The model processes audio features extracted using `wav2vec2` and predicts the emotion category. The project includes:
- Data preprocessing and feature extraction.
- A custom deep learning model with convolutional, recurrent, and attention layers.
- Training and evaluation scripts.
- Saved models for reuse.

---

## Datasets

The following datasets were used for training and evaluation:
1. **RAVDESS**: [RAVDESS Dataset]
2. **SAVEE**: [SAVEE Dataset]
3. **TESS**: [TESS Dataset]
4. **CREMA-D**: [CREMA-D Dataset]

The datasets were combined and preprocessed to create a unified dataset for training.

---

## Model Architecture

The model is built using TensorFlow/Keras and includes the following components:
1. **Convolutional Layers**: Extract spatial features from audio data.
2. **Recurrent Layers (BiLSTM)**: Capture temporal dependencies in the audio signals.
3. **Attention Mechanism**: Focus on important parts of the audio sequence.
4. **Fully Connected Layers**: Perform classification into emotion categories.

Key features:
- Residual connections for better gradient flow.
- Multi-head attention for enhanced feature extraction.
- Focal loss for handling class imbalance.

## Usage
1. **Preprocess Data**
Run the script to preprocess and combine datasets:
python [pathsave.py]

2. **Train the Model**
Train the emotion recognition model:
python [train_emotion_model.py]

3. **Evaluate the Model**
Evaluate the trained model on the test set:
python evaluate_model.py

4. **Use the Model**
Use the saved model for inference:

model = load_model("final_emotion_model.keras")
Pass your audio features to the model for predictions

## Acknowledgments
This project was created by **Nikesh**. The model was built from scratch and trained using publicly available datasets. Special thanks to the creators of the RAVDESS, SAVEE, TESS, and CREMA-D datasets for providing the data used in this project.

## Contact
nikeshkumar31030@gmail.com

