# models/biomedical_model.py

import os
import pickle
import pandas as pd
import numpy as np

# Load the trained biomedical model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "biomedical_model.pkl")

with open(MODEL_PATH, 'rb') as f:
    allWrist = pickle.load(f)

# Define the order of features
feature_names = ['wax', 'way', 'waz', 'wbvp', 'weda', 'wtemp']

def predict_stress_score(feature_list):
    """
    Input: list of 6 features in this order:
           [wax, way, waz, wbvp, weda, wtemp]
    Output: stress_score (float between 0 and 1)
    """
    input_df = pd.DataFrame([feature_list], columns=feature_names)
    probs = allWrist.predict_proba(input_df)[0]
    label = np.argmax(probs)
    stress_prob = probs[1]

    return round(stress_prob, 4)
