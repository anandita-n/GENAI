import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
APP_DIR = BASE_DIR / "app"
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
STATIC_DIR = APP_DIR / "static"

# Ensure directories exist
DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Model Artifacts
HAND_LANDMARKER_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"
CLASSIFIER_MODEL_PATH = MODELS_DIR / "asl_classifier.joblib"
LABEL_ENCODER_PATH = MODELS_DIR / "label_encoder.joblib"
SCALER_PATH = MODELS_DIR / "scaler.joblib"
METRICS_PATH = DATA_DIR / "eval_metrics.json"

# Gesture Recognition Settings
DEFAULT_CONFIDENCE_THRESHOLD = 0.35  # Sensitive threshold (29 classes, random chance is 0.034)
DEFAULT_STABILITY_FRAMES = 7        # ~0.45s hold delay before committing letter
COOLDOWN_SECONDS = 2.0              # Lockout delay to prevent duplicate repeats when holding gesture

# LLM Configuration
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

# ASL Class Definitions
ASL_ALPHABET = [chr(i) for i in range(ord('A'), ord('Z') + 1)]
SPECIAL_GESTURES = ["SPACE", "DEL", "CLEAR"]
ALL_CLASSES = ASL_ALPHABET + SPECIAL_GESTURES

# Server Config
HOST = os.getenv("HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8000"))
