"""
Layer 4 – Pseudo-Depth / Geometry Integrity
Purpose: Detect face warping, landmark distortion, unnatural proportions.
Tech Stack: MediaPipe Tasks FaceLandmarker (landmarks) + NumPy geometry ratios
Output: L4 geometry distortion score + ratio drift reason + status
"""
import cv2
import numpy as np
from collections import deque
from typing import Optional

from app.core.types import FrameData
from app.core import config
from app.modules.face_detector import FaceDetector


class GeometryIntegrityChecker:
    """
    L4: Pseudo-Depth / Geometry Integrity
    Analyses facial landmark geometry ratios to detect:
    - Face warping (deepfake morphing)
    - Unnatural facial proportions (printed photo / 3D mask)
    - Symmetry anomalies
    - Landmark drift across frames

    Requires MediaPipe Tasks FaceLandmarker; reports UNAVAILABLE when it is
    not initialized (never fabricates a geometry prediction).
    """

    SYMMETRY_THRESHOLD = 0.25  # Asymmetry ratio above this -> suspicious
    RATIO_DRIFT_THRESH = 0.15  # Ratio drift above this -> warping
    HISTORY_LEN = 20

    def __init__(self, face_detector: Optional[FaceDetector] = None):
        self.face_detector = face_detector
        self.use_mediapipe = bool(
            face_detector is not None and face_detector.use_mediapipe
        )

        self.ratio_history: deque = deque(maxlen=self.HISTORY_LEN)
        self.status = (
            config.LAYER_STATUS_UNAVAILABLE
            if not self.use_mediapipe
            else config.LAYER_STATUS_PASS
        )
        if not self.use_mediapipe:
            print("L4-Geometry: MediaPipe FaceLandmarker unavailable — "
                  "reporting UNAVAILABLE")

    # ── Public API ────────────────────────────────────────────────────────
    def process(self, frame_data: FrameData) -> None:
        frame = frame_data.image
        if frame is None or frame.size == 0:
            frame_data.metadata["l4_geometry_score"] = None
            frame_data.metadata["l4_status"] = config.LAYER_STATUS_ERROR
            self.status = config.LAYER_STATUS_ERROR
            return

        if not self.use_mediapipe or self.face_detector is None:
            frame_data.metadata["l4_geometry_score"] = None
            frame_data.metadata["l4_status"] = config.LAYER_STATUS_UNAVAILABLE
            self.status = config.LAYER_STATUS_UNAVAILABLE
            return

        h, w = frame.shape[:2]
        landmarks = self.face_detector.landmarks(frame)
        if landmarks is None:
            frame_data.metadata["l4_geometry_score"] = None
            frame_data.flags.append("L4_NO_LANDMARKS")
            frame_data.metadata["l4_status"] = config.LAYER_STATUS_INCONCLUSIVE
            self.status = config.LAYER_STATUS_INCONCLUSIVE
            return

        def pt(idx):
            return np.array([landmarks[idx][0] * w, landmarks[idx][1] * h])

        score = 1.0
        reasons = []

        # ── 1. Facial proportion ratios ───────────────────────────────────
        left_eye_w = float(np.linalg.norm(pt(33) - pt(133)))
        right_eye_w = float(np.linalg.norm(pt(362) - pt(263)))
        eye_ratio = left_eye_w / (right_eye_w + 1e-6)

        nose_tip = pt(1)
        chin = pt(152)
        left_eye_c = (pt(33) + pt(133)) / 2
        right_eye_c = (pt(362) + pt(263)) / 2
        eye_center = (left_eye_c + right_eye_c) / 2

        eye_nose_dist = float(np.linalg.norm(eye_center - nose_tip))
        nose_chin_dist = float(np.linalg.norm(nose_tip - chin))
        proportion_ratio = eye_nose_dist / (nose_chin_dist + 1e-6)

        # ── 2. Symmetry check ─────────────────────────────────────────────
        face_center_x = (pt(33)[0] + pt(263)[0]) / 2
        left_dist = abs(pt(33)[0] - face_center_x)
        right_dist = abs(pt(263)[0] - face_center_x)
        symmetry_ratio = abs(left_dist - right_dist) / (left_dist + right_dist + 1e-6)

        if symmetry_ratio > self.SYMMETRY_THRESHOLD:
            score -= 0.25
            reasons.append("L4_FACE_ASYMMETRY")

        # ── 3. Proportion drift across frames ─────────────────────────────
        current_ratios = np.array([eye_ratio, proportion_ratio])
        drift = 0.0
        if len(self.ratio_history) >= 5:
            hist_mean = np.mean(list(self.ratio_history), axis=0)
            drift = float(np.linalg.norm(current_ratios - hist_mean))
            if drift > self.RATIO_DRIFT_THRESH:
                score -= 0.30
                reasons.append("L4_LANDMARK_RATIO_DRIFT")
            frame_data.metadata["l4_ratio_drift"] = round(drift, 4)

        self.ratio_history.append(current_ratios)

        # ── 4. Unnatural eye ratio (warped face) ──────────────────────────
        if eye_ratio < 0.7 or eye_ratio > 1.4:
            score -= 0.20
            reasons.append("L4_EYE_WIDTH_DISTORTION")

        score = float(np.clip(score, 0.0, 1.0))
        frame_data.metadata["l4_geometry_score"] = round(score, 4)
        frame_data.metadata["l4_eye_ratio"] = round(float(eye_ratio), 4)
        frame_data.metadata["l4_symmetry"] = round(float(symmetry_ratio), 4)
        frame_data.metadata["l4_status"] = (
            config.LAYER_STATUS_PASS if not reasons else config.LAYER_STATUS_FAIL
        )
        self.status = config.LAYER_STATUS_PASS
        frame_data.flags.extend(reasons)
