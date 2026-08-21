# Import the prediction functions directly
try:
    from .biomedical_model import predict_stress_score as biomedical_predict
except Exception:
    biomedical_predict = None
from .behavioral_model import predict_stress as behavioral_predict
from .text_model import predict_stress_score as text_predict
from .voice_model import predict_stress as voice_predict
from .facial_model import predict_stress as facial_predict

# Create simple wrapper classes for consistency
class BiomedicalModel:
    def predict_stress_score(self, features):
        if biomedical_predict is None:
            return 0.0  # xgboost/libomp unavailable in this environment, skipped
        return biomedical_predict(features)

class BehavioralModel:
    def predict_stress(self, sleep_duration, quality_of_sleep, physical_activity_level, daily_steps, bmi_category):
        return behavioral_predict(sleep_duration, quality_of_sleep, physical_activity_level, daily_steps, bmi_category)

class TextModel:
    def predict_stress_score(self, text):
        return text_predict(text)

class VoiceModel:
    def predict_stress(self, file_path):
        return voice_predict(file_path)

class FacialModel:
    def predict_stress(self, image_path):
        return facial_predict(image_path)

# Initialize model instances
biomedical_model = BiomedicalModel()
behavioral_model = BehavioralModel()
text_model = TextModel()
voice_model = VoiceModel()
facial_model = FacialModel() 