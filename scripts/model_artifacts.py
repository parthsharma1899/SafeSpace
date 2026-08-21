from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "safespace_fusion" / "models"

ARTIFACT_PATTERNS = [
    "biomedical_model.pkl",
    "random_forest_stress_model.pkl",
    "speech_model.keras",
    "speech_model.h5",
    "elm_weights-001.npz",
    "features/**",
    "stress_classifier_epoch2/**",
    "models_face/resnet50_features/**",
    "models_face/vit_features/**",
    "models_face/stress1.jpg",
    "models_face/stress2.jpg",
    "parth.wav",
]

IGNORED_UPLOAD_PATTERNS = [
    "**/.DS_Store",
    "**/__pycache__/**",
    "**/.ipynb_checkpoints/**",
]

REQUIRED_FILES = [
    "biomedical_model.pkl",
    "random_forest_stress_model.pkl",
    "speech_model.keras",
    "elm_weights-001.npz",
    "features/scaler_mean.npy",
    "features/scaler_scale.npy",
    "stress_classifier_epoch2/config.json",
    "stress_classifier_epoch2/model.safetensors",
    "stress_classifier_epoch2/tokenizer_config.json",
    "stress_classifier_epoch2/special_tokens_map.json",
    "stress_classifier_epoch2/vocab.txt",
    "models_face/resnet50_features/saved_model.pb",
    "models_face/resnet50_features/variables/variables.data-00000-of-00001",
    "models_face/resnet50_features/variables/variables.index",
    "models_face/vit_features/saved_model.pb",
    "models_face/vit_features/variables/variables.data-00000-of-00001",
    "models_face/vit_features/variables/variables.index",
    "models_face/stress1.jpg",
    "parth.wav",
]


def missing_required_files(models_dir=MODELS_DIR):
    models_dir = Path(models_dir)
    return [path for path in REQUIRED_FILES if not (models_dir / path).exists()]
