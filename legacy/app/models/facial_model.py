# facial_model.py

import numpy as np
import tensorflow as tf
import cv2
import os
from tensorflow.keras.preprocessing import image as keras_image

# ========== Path Configuration ==========
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ELM_PATH = os.path.join(BASE_DIR, "elm_weights-001.npz")
SCALER_MEAN_PATH = os.path.join(BASE_DIR, "features", "scaler_mean.npy")
SCALER_SCALE_PATH = os.path.join(BASE_DIR, "features", "scaler_scale.npy")
RESNET_MODEL_PATH = os.path.join(BASE_DIR, "models_face", "resnet50_features")
VIT_MODEL_PATH = os.path.join(BASE_DIR, "models_face", "vit_features")

# ========== Load Components ==========
def load_components():
    try:
        # ELM Weights
        elm_weights = np.load(ELM_PATH)
        W = elm_weights['W']
        b = elm_weights['b']
        B = elm_weights['B']

        # Scalers
        scaler_mean = np.load(SCALER_MEAN_PATH)
        scaler_scale = np.load(SCALER_SCALE_PATH)
        scaler_scale[scaler_scale == 0] = 1.0  # Prevent division by zero

        # Models
        resnet_model = tf.saved_model.load(RESNET_MODEL_PATH)
        vit_model = tf.saved_model.load(VIT_MODEL_PATH)

        if 'serving_default' not in resnet_model.signatures:
            raise ValueError("ResNet model missing 'serving_default' signature")
        if 'serving_default' not in vit_model.signatures:
            raise ValueError("ViT model missing 'serving_default' signature")

        return {
            'W': W, 'b': b, 'B': B,
            'scaler_mean': scaler_mean,
            'scaler_scale': scaler_scale,
            'resnet_model': resnet_model,
            'vit_model': vit_model,
            'face_cascade': cv2.CascadeClassifier(
                cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            )
        }

    except Exception as e:
        raise RuntimeError(f"Component loading failed: {str(e)}")

# ========== Utility ==========
def safe_divide(a, b):
    return np.divide(a, b, out=np.zeros_like(a), where=b != 0)

# ========== Feature Extraction ==========
def extract_features(face_img, models):
    try:
        img = cv2.resize(face_img, (224, 224))
        img_tensor = tf.convert_to_tensor(
            np.expand_dims(keras_image.img_to_array(img) / 255.0, axis=0),
            dtype=tf.float32
        )

        resnet_out = models['resnet_model'].signatures['serving_default'](img_tensor)
        vit_out = models['vit_model'].signatures['serving_default'](img_tensor)

        resnet_features = next(iter(resnet_out.values())).numpy().flatten()
        vit_features = next(iter(vit_out.values())).numpy().flatten()

        return np.concatenate([resnet_features, vit_features])

    except Exception as e:
        raise RuntimeError(f"Feature extraction failed: {str(e)}")

# ========== Main Prediction ==========
def predict_stress(image_path):
    try:
        components = load_components()

        img = cv2.imread(image_path)
        if img is None:
            return "Error: Invalid image file"

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = components['face_cascade'].detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5
        )

        if len(faces) == 0:
            return "Error: No face detected"

        x, y, w, h = faces[0]
        face_img = img[y:y+h, x:x+w]

        features = extract_features(face_img, components)

        features_norm = safe_divide(
            features - components['scaler_mean'],
            components['scaler_scale']
        )

        H = np.maximum(
            0.01 * (features_norm @ components['W'] + components['b']),
            features_norm @ components['W'] + components['b']
        )
        stress_score = (H @ components['B']).item()

        return float(np.clip(stress_score, 0, 1))

    except Exception as e:
        return f"Error: {str(e)}"
