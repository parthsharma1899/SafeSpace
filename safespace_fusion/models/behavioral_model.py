# behavioral_model.py

import pandas as pd

import os

import joblib


MODEL_PATH = os.path.join(os.path.dirname(__file__), "random_forest_stress_model.pkl")
model = joblib.load(MODEL_PATH)

def predict_stress(sleep_duration, quality_of_sleep, physical_activity_level, daily_steps, bmi_category):
    input_df = pd.DataFrame([[
        sleep_duration,
        physical_activity_level,
        daily_steps,
        quality_of_sleep,
        bmi_category
    ]], columns=[
        'Sleep Duration',
        'Physical Activity Level',
        'Daily Steps',
        'Quality of Sleep',
        'BMI Category'
    ])

    pred_scaled = model.predict(input_df)[0]
    return round(pred_scaled, 3)
