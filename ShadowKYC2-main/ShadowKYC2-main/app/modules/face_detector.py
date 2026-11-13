import cv2
import numpy as np
import time
from typing import Optional, Tuple, List

from app.core import config

# MediaPipe Tasks (FaceLandmarker) is used for facial landmarks.
# The legacy `mp.solutions` API is NOT available in MediaPipe 1.x (Tasks-only
# build), so we do not import `mediapipe.solutions` anywhere.
_HAS_MEDIAPIPE_TASKS = False
try:
    import mediapipe as mp
    from mediapipe.tasks import python as mp_tasks
    from mediapipe.tasks.python import vision as mp_vision
    _HAS_MEDIAPIPE_TASKS = True
except Exception as _e:  # pragma: no cover - environment dependent
    mp = None  # type: ignore
    mp_tasks = None  # type: ignore
    mp_vision = None  # type: ignore
    _LANDMARKER_IMPORT_ERROR = _e
else:
    _LANDMARKER_IMPORT_ERROR = None


class LandmarkDetector:
    """
    MediaPipe Tasks FaceLandmarker wrapper.

    Returns normalized facial landmarks (468 face + optional iris points) or
    None when no face is found or the detector is unavailable.
    """

    def __init__(self, model_path: Optional[str] = None,
                 running_mode: Optional[object] = None):
        """
        MediaPipe Tasks FaceLandmarker wrapper.

        ``running_mode`` defaults to IMAGE (used by the passive L1/L4/L5 layers
        which process single frames). LiveGuard active challenges pass
        RunningMode.VIDEO so the landmarker can track across frames.
        The same class is reused for both modes — no duplicate detector.
        """
        self.available = False
        self.landmarker = None
        self.init_error: Optional[str] = None
        self.running_mode = running_mode
        if not _HAS_MEDIAPIPE_TASKS:
            self.init_error = f"MediaPipe Tasks not installed ({_LANDMARKER_IMPORT_ERROR})"
            print(f"[FaceDetector] LandmarkDetector unavailable: {self.init_error}")
            return
        if running_mode is None:
            running_mode = mp_vision.RunningMode.IMAGE
        self.running_mode = running_mode
        path = model_path or str(config.FACE_LANDMARK_MODEL_PATH)
        try:
            base_options = mp_tasks.BaseOptions(model_asset_path=path)
            options = mp_vision.FaceLandmarkerOptions(
                base_options=base_options,
                running_mode=running_mode,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            self.landmarker = mp_vision.FaceLandmarker.create_from_options(options)
            self.available = True
            mode_name = getattr(running_mode, "name", str(running_mode))
            print(f"[FaceDetector] MediaPipe Tasks FaceLandmarker ready ({mode_name})")
        except Exception as e:  # model missing, download blocked, etc.
            self.available = False
            self.landmarker = None
            self.init_error = str(e)
            print(f"[FaceDetector] MediaPipe Tasks FaceLandmarker unavailable ({e}). "
                  f"Landmark-based layers will report UNAVAILABLE.")

    def detect(self, frame_bgr: np.ndarray,
               timestamp_ms: Optional[int] = None) -> Optional[List[Tuple[float, float]]]:
        """
        Detect facial landmarks on a BGR image.
        Returns a list of (x, y) normalized coordinates (0..1) or None.

        For VIDEO mode callers, pass ``timestamp_ms`` (monotonic ms) so the
        landmarker can track across frames.
        """
        if not self.available or self.landmarker is None:
            return None
        if frame_bgr is None or frame_bgr.size == 0:
            return None
        try:
            rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            if self.running_mode == mp_vision.RunningMode.VIDEO:
                if timestamp_ms is None:
                    timestamp_ms = int(time.time() * 1000)
                result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
            else:
                result = self.landmarker.detect(mp_image)
            if not result.face_landmarks:
                return None
            landmarks = result.face_landmarks[0]
            return [(float(p.x), float(p.y)) for p in landmarks]
        except Exception as e:
            print(f"[FaceDetector] Landmark detection error: {e}")
            return None


class FaceDetector:
    """
    Face detection wrapper.

    Detection (bounding box + ROI) is performed with OpenCV Haar Cascades,
    which is reliable and works offline. Facial landmarks (for L1/L4/L5) are
    provided by the optional MediaPipe Tasks FaceLandmarker; when it is not
    available those layers report UNAVAILABLE instead of fabricating output.
    """

    def __init__(self, min_detection_confidence: float = 0.5,
                 running_mode: Optional[object] = None):
        self.min_detection_confidence = min_detection_confidence
        self.target_size = (224, 224)
        self.running_mode = running_mode

        # ── OpenCV Haar Cascade (detection / ROI) ──────────────────────────
        self.haar_available = False
        self.haar_error: Optional[str] = None
        try:
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise RuntimeError("Haar Cascade classifier is empty")
            self.haar_available = True
            print(f"[FaceDetector] OpenCV Haar Cascade ready ({cascade_path})")
        except Exception as e:
            self.haar_available = False
            self.face_cascade = None
            self.haar_error = str(e)
            print(f"[FaceDetector] ERROR: OpenCV Haar Cascade unavailable ({e})")

        # ── MediaPipe Tasks FaceLandmarker (landmarks only) ────────────────
        self.landmark_detector = LandmarkDetector(running_mode=running_mode)
        self.use_mediapipe = self.landmark_detector.available

    # ── Public API ────────────────────────────────────────────────────────
    def process_frame(
        self, frame: np.ndarray
    ) -> Tuple[Optional[np.ndarray], Optional[Tuple[int, int, int, int]]]:
        """
        Detect the largest face and return a resized ROI + bounding box.
        Uses OpenCV Haar Cascades (offline, reliable).
        """
        if not self.haar_available or self.face_cascade is None:
            return None, None
        if frame is None or frame.size == 0:
            return None, None

        try:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
            if len(faces) == 0:
                return None, None

            # Sort by area and take the largest face
            faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
            x, y, w, h = [int(v) for v in faces[0]]
            return self._extract_roi(frame, x, y, w, h), (x, y, w, h)
        except Exception as e:
            print(f"[FaceDetector] Haar detection error: {e}")
            return None, None

    def landmarks(self, frame_bgr: np.ndarray) -> Optional[List[Tuple[float, float]]]:
        """Return normalized facial landmarks or None (MediaPipe Tasks)."""
        return self.landmark_detector.detect(frame_bgr)

    def _extract_roi(
        self, frame, x: int, y: int, width: int, height: int
    ) -> Optional[np.ndarray]:
        h, w = frame.shape[:2]
        x = max(0, x)
        y = max(0, y)
        width = min(w - x, width)
        height = min(h - y, height)
        if width <= 0 or height <= 0:
            return None
        roi = frame[y:y + height, x:x + width]
        try:
            return cv2.resize(roi, self.target_size)
        except Exception as e:
            print(f"[FaceDetector] Error resizing ROI: {e}")
            return None

    def diagnostics(self) -> dict:
        """Startup diagnostic report for every detector."""
        return {
            "haar_cascade": {
                "available": self.haar_available,
                "error": self.haar_error,
            },
            "mediapipe_tasks": {
                "available": _HAS_MEDIAPIPE_TASKS,
                "version": getattr(mp, "__version__", None) if mp else None,
                "note": "legacy mp.solutions API not used (MediaPipe 1.x Tasks-only build)",
            },
            "face_landmarker": {
                "available": self.landmark_detector.available,
                "model_path": str(config.FACE_LANDMARK_MODEL_PATH),
                "model_exists": config.FACE_LANDMARK_MODEL_PATH.exists(),
                "error": self.landmark_detector.init_error,
            },
        }
