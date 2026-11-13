"""
Layer 5 – Lip-Sync Integrity Validation
Purpose: Detect mismatch between voice and mouth movement.
Tech Stack: MediaPipe Tasks FaceLandmarker (landmarks) + NumPy
Output: L5 lip-sync score + mouth-motion reason + status
"""
import cv2
import numpy as np
from collections import deque
from typing import Optional

from app.core.types import FrameData
from app.core import config
from app.modules.face_detector import FaceDetector


class LipSyncValidator:
    """
    L5: Lip-Sync Integrity Validation

    Tracks mouth aperture (MAR – Mouth Aspect Ratio) over time. In a live
    session without audio access, MAR variance is used as a proxy: a real
    speaker has natural mouth movement variation. A frozen mouth while the
    session is live is suspicious.

    When audio chunks are provided, audio RMS energy is correlated with MAR.

    Requires MediaPipe Tasks FaceLandmarker; reports UNAVAILABLE when it is
    not initialized (never fabricates a lip-sync prediction).
    """

    MAR_FROZEN_THRESHOLD = 0.0005  # Variance below -> mouth not moving
    MAR_ERRATIC_THRESHOLD = 0.08   # Variance above -> erratic/unnatural
    HISTORY_LEN = 30
    MIN_FRAMES_FOR_EVAL = 10

    def __init__(self, face_detector: Optional[FaceDetector] = None):
        self.face_detector = face_detector
        self.use_mediapipe = bool(
            face_detector is not None and face_detector.use_mediapipe
        )

        self.mar_history: deque = deque(maxlen=self.HISTORY_LEN)
        self.audio_rms_history: deque = deque(maxlen=self.HISTORY_LEN)

        self.status = (
            config.LAYER_STATUS_UNAVAILABLE
            if not self.use_mediapipe
            else config.LAYER_STATUS_PASS
        )
        if not self.use_mediapipe:
            print("L5-LipSync: MediaPipe FaceLandmarker unavailable — "
                  "reporting UNAVAILABLE")

    # ── Public API ────────────────────────────────────────────────────────
    def process(self, frame_data: FrameData, audio_rms: Optional[float] = None) -> None:
        """
        Args:
            frame_data: current frame data
            audio_rms:  optional RMS energy of the corresponding audio chunk
        """
        frame = frame_data.image
        if frame is None or frame.size == 0:
            frame_data.metadata["l5_lipsync_score"] = None
            frame_data.metadata["l5_status"] = config.LAYER_STATUS_ERROR
            self.status = config.LAYER_STATUS_ERROR
            return

        if not self.use_mediapipe or self.face_detector is None:
            frame_data.metadata["l5_lipsync_score"] = None
            frame_data.metadata["l5_status"] = config.LAYER_STATUS_UNAVAILABLE
            self.status = config.LAYER_STATUS_UNAVAILABLE
            return

        landmarks = self.face_detector.landmarks(frame)
        if landmarks is None:
            frame_data.metadata["l5_lipsync_score"] = None
            frame_data.metadata["l5_status"] = config.LAYER_STATUS_INCONCLUSIVE
            self.status = config.LAYER_STATUS_INCONCLUSIVE
            return

        mar = self._compute_mar(landmarks)
        self.mar_history.append(mar)

        if audio_rms is not None:
            self.audio_rms_history.append(audio_rms)

        score = 1.0
        reasons = []
        mar_var = None

        if len(self.mar_history) >= self.MIN_FRAMES_FOR_EVAL:
            mar_arr = np.array(list(self.mar_history))
            mar_var = float(np.var(mar_arr))

            # In silent video KYC, natural resting mouth is 100% expected.
            # Only check mismatch if speech audio is actually present.
            if audio_rms is not None and audio_rms > 0.03:
                if mar_var < self.MAR_FROZEN_THRESHOLD:
                    score -= 0.40
                    reasons.append("L5_MOUTH_FROZEN_DURING_SPEECH")
                elif mar_var > self.MAR_ERRATIC_THRESHOLD:
                    score -= 0.25
                    reasons.append("L5_ERRATIC_MOUTH_MOVEMENT")

            if audio_rms is not None and len(self.audio_rms_history) >= self.MIN_FRAMES_FOR_EVAL:
                audio_arr = np.array(list(self.audio_rms_history))
                n = min(len(mar_arr), len(audio_arr))
                if audio_arr.std() > 1e-6 and mar_arr.std() > 1e-6:
                    corr = float(np.corrcoef(mar_arr[-n:], audio_arr[:n])[0, 1])
                    frame_data.metadata["l5_audio_mar_corr"] = round(corr, 4)
                    if corr < 0.15:
                        score -= 0.30
                        reasons.append("L5_AUDIO_LIP_MISMATCH")

        if mar_var is not None:
            frame_data.metadata["l5_mar_var"] = round(mar_var, 6)

        score = float(np.clip(score, 0.0, 1.0))
        frame_data.metadata["l5_lipsync_score"] = round(score, 4)
        frame_data.metadata["l5_mar"] = round(float(mar), 4)
        frame_data.metadata["l5_status"] = (
            config.LAYER_STATUS_PASS if not reasons else config.LAYER_STATUS_FAIL
        )
        self.status = config.LAYER_STATUS_PASS
        frame_data.flags.extend(reasons)

    # ── Helpers ───────────────────────────────────────────────────────────
    @staticmethod
    def _compute_mar(landmarks) -> float:
        """Mouth Aspect Ratio using MediaPipe-style normalized landmarks."""
        from scipy.spatial import distance as dist
        # Vertical: top lip (13) to bottom lip (14)
        # Horizontal: left corner (78) to right corner (308)
        top = (landmarks[13][0], landmarks[13][1])
        bottom = (landmarks[14][0], landmarks[14][1])
        left = (landmarks[78][0], landmarks[78][1])
        right = (landmarks[308][0], landmarks[308][1])
        vertical = dist.euclidean(top, bottom)
        horizontal = dist.euclidean(left, right)
        return float(vertical / (horizontal + 1e-6))
