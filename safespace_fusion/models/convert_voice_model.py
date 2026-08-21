# convert_voice_model.py

from tensorflow.keras.models import load_model

# Load the old .h5 model
old_model = load_model("speech_model.h5")

# Save in new format
old_model.save("speech_model.keras", save_format="keras")  # ✅ New format
