from flask import Flask, render_template, request, jsonify
import os
import time
from pathlib import Path
from fusion_engine import FusionEngine
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = str(BASE_DIR / 'uploads')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize Fusion Engine
fusion = FusionEngine()

# Allowed file extensions
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'jpg', 'jpeg', 'png'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_models():
    """Load models only when needed"""
    from models import (
        biomedical_model,
        behavioral_model,
        text_model,
        voice_model,
        facial_model
    )
    return biomedical_model, behavioral_model, text_model, voice_model, facial_model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        # Load models only when needed
        biomedical_model, behavioral_model, text_model, voice_model, facial_model = load_models()
        
        # Get form data
        input_text = request.form.get('input_text', '')
        
        # Biomedical features
        biomedical_features = [
            float(request.form.get('wax', 0)),
            float(request.form.get('way', 0)),
            float(request.form.get('waz', 0)),
            float(request.form.get('bvp', 0)),
            float(request.form.get('eda', 0)),
            float(request.form.get('temp', 0))
        ]
        
        # Behavioral data
        sleep_duration = float(request.form.get('sleep_duration', 7))
        quality_of_sleep = float(request.form.get('quality_of_sleep', 6))
        physical_activity_level = float(request.form.get('physical_activity_level', 45))
        daily_steps = float(request.form.get('daily_steps', 7000))
        bmi_category = int(request.form.get('bmi_category', 0))
        
        # Handle file uploads
        voice_file_path = None
        face_image_path = None
        
        # Voice file upload
        if 'voice_file' in request.files:
            voice_file = request.files['voice_file']
            if voice_file and voice_file.filename != '' and allowed_file(voice_file.filename):
                filename = secure_filename(voice_file.filename)
                voice_file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                voice_file.save(voice_file_path)
        
        # Face image upload
        if 'face_image' in request.files:
            face_image = request.files['face_image']
            if face_image and face_image.filename != '' and allowed_file(face_image.filename):
                filename = secure_filename(face_image.filename)
                face_image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                face_image.save(face_image_path)
        
        # Use default files if none uploaded
        if not voice_file_path:
            voice_file_path = str(BASE_DIR / "models" / "parth.wav")
        if not face_image_path:
            face_image_path = str(BASE_DIR / "models" / "models_face" / "stress1.jpg")
        
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
        
        # Prepare response
        result = results[0]
        
        # Convert all values to native Python types for JSON serialization
        def convert_to_native(obj):
            if hasattr(obj, 'item'):  # NumPy scalar
                return obj.item()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_native(v) for v in obj]
            elif isinstance(obj, (int, float, str, bool)) or obj is None:
                return obj
            else:
                return str(obj)
        
        response = {
            'success': True,
            'wmss': convert_to_native(result['WMSS']),
            'stress_level': result['Stress Level'],
            'dominant_modality': result['Dominant Modality'],
            'normalized_weights': convert_to_native(result['Normalized Weights']),
            'individual_scores': convert_to_native(sample_scores),
            'fusion_details': {
                'ema': convert_to_native({mod: round(fusion.ema.get(mod, 0), 4) for mod in fusion.modalities}),
                'residual': convert_to_native({mod: round(fusion.residual.get(mod, 0), 4) for mod in fusion.modalities}),
                'persistence': convert_to_native({mod: round(fusion.persistence.get(mod, 0), 4) for mod in fusion.modalities})
            }
        }
        
        return jsonify(response)
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080) 
