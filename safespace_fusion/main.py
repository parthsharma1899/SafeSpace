import os
import time
import random
from fusion_engine import FusionEngine
from models import (
    biomedical_model,
    behavioral_model,
    text_model,
    voice_model,
    facial_model
)

# Initialize Fusion Engine
fusion = FusionEngine()
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Simulated input text samples
texts = [
    "I'm okay.",
    "I'm feeling slightly anxious today.",
    "I couldn't sleep well and feel on edge.",
    "I'm really overwhelmed by all the tasks.",
    "Today was calm, no stress at all."
]

# Image path for facial input
face_image_path = os.path.join(BASE_DIR, "models", "models_face", "stress1.jpg")

# Run for 5 time steps
for t in range(5):
    print(f"\n--- TIME STEP {t + 1} ---")

    # Simulate varying input text
    input_text = texts[t]

    # Simulate biomedical features
    biomedical_features = [
        random.uniform(-35, -30),     # wax
        random.uniform(-60, -50),     # way
        random.uniform(3, 6),         # waz
        random.uniform(-0.9, -0.7),   # bvp
        random.uniform(0.3, 0.5),     # eda
        random.uniform(31, 33)        # temp
    ]

    # Behavioral simulated values
    sleep_duration = 7
    quality_of_sleep = 6 + (t % 2)
    physical_activity_level = 45
    daily_steps = 7000 + 100 * t
    bmi_category = 0

    # Voice file (same for now)
    voice_file_path = os.path.join(BASE_DIR, "models", "parth.wav")

    # Collect predictions from all modalities
    sample_scores = {
        'biomedical': biomedical_model.predict_stress_score(biomedical_features),
        'behavioral': behavioral_model.predict_stress(sleep_duration, quality_of_sleep, physical_activity_level, daily_steps, bmi_category),
        'text': text_model.predict_stress_score(input_text),
        'voice': voice_model.predict_stress(voice_file_path),
        'facial': facial_model.predict_stress(face_image_path)
    }

    # Run through fusion engine
    batch = [sample_scores]
    results = fusion.run_batch(batch)

    # Output results
    for result in results:
        print("WMSS:", result['WMSS'])
        print("Stress Level:", result['Stress Level'])
        print("Dominant Modality:", result['Dominant Modality'])
        print("Normalized Weights:")
        for modality, weight in result['Normalized Weights'].items():
            print(f"  {modality}: {weight:.4f}")

        # Print internal fusion state
        print("\n--- Residual-Persistence Details ---")
        for modality in sample_scores.keys():
            ema = fusion.ema.get(modality, 0)
            residual = fusion.residual.get(modality, 0)
            persistence = fusion.persistence.get(modality, 0)
            print(f"{modality.upper()} -> EMA: {ema:.4f} | Residual: {residual:.4f} | Persistence: {persistence:.4f}")

    time.sleep(1)
