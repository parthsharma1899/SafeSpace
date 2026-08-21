# models/voice_model.py

import os
import numpy as np
import librosa
from tensorflow.keras.models import load_model

MODEL_PATH = os.path.join(os.path.dirname(__file__), "speech_model.keras")
model = load_model(MODEL_PATH)

# Feature extraction
def extract_features(file_path, max_len=216):
    audio, sr = librosa.load(file_path, duration=3, offset=0.5)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
    padded = np.pad(mfccs, ((0, 0), (0, max(0, max_len - mfccs.shape[1]))), mode='constant')
    return padded[:, :max_len]

# Prediction
def predict_stress(file_path):
    """
    Takes a path to a .wav file and returns a stress score between 0 and 1.
    """
    features = extract_features(file_path)
    features = np.expand_dims(features, axis=-1)  # (40, 216, 1)
    features = np.expand_dims(features, axis=0)   # (1, 40, 216, 1)

    score = model.predict(features)[0][0]
    return round(float(score), 4)
