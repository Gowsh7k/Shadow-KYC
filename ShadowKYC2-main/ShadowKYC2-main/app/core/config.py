from pathlib import Path

# --- Thresholds ---
BLUR_THRESHOLD = 100.0  # Laplacian variance below this -> blur flag
BRIGHTNESS_THRESHOLD = 50.0  # Mean brightness below this -> dark frame
LIVENESS_BLINK_MIN = 3.0  # Minimum blinks per minute for normal behavior
TEMPORAL_HASH_SIM_THRESHOLD = 0.85  # Repeating hash similarity for loop detection
DEEPFAKE_EXEC_THRESHOLD = 0.35  # Layer 5 execution threshold
RISK_LOW_THRESHOLD = 0.3  # Risk classification cutoff
RISK_HIGH_THRESHOLD = 0.6  # Risk classification cutoff

# --- Layer Weights ---
LAYER_WEIGHTS = {
    "quality": 0.1,
    "artifact": 0.15,
    "liveness": 0.3,
    "temporal": 0.25,
    "deepfake": 0.2
}

# --- Paths ---
BASE_DIR = Path(__file__).resolve().parent.parent.parent
EVIDENCE_DIR = BASE_DIR / "evidence"
REPORTS_DIR = BASE_DIR / "reports"
MODELS_DIR = BASE_DIR / "app" / "models"

# --- Model Paths ---
DEEPFAKE_MODEL_PATH = MODELS_DIR / "deepfake_model.tflite"  # Placeholder path (no bundled deepfake classifier)
FACE_LANDMARK_MODEL_PATH = MODELS_DIR / "face_landmarker.task"  # MediaPipe Tasks FaceLandmarker bundle
MINIFASNET_MODEL_PATH = MODELS_DIR / "minifasnet_v2.onnx"  # MiniFASNet V2 deep anti-spoofing classifier

# --- Result statuses (used by every verification layer) ---
# PASS   - check ran and signal is within acceptable range
# FAIL   - check ran and signal is suspicious
# INCONCLUSIVE - check ran but could not reach a confident conclusion
# UNAVAILABLE - model/detector missing; no prediction made (never treated as fraud)
# ERROR  - check raised an exception
LAYER_STATUS_PASS = "PASS"
LAYER_STATUS_FAIL = "FAIL"
LAYER_STATUS_INCONCLUSIVE = "INCONCLUSIVE"
LAYER_STATUS_UNAVAILABLE = "UNAVAILABLE"
LAYER_STATUS_ERROR = "ERROR"

# Make sure directories exist
EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)
