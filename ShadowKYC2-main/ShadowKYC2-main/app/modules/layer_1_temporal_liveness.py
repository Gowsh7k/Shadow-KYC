"""
Layer 1 – Temporal Liveness Integrity
Purpose: Detect unnatural stability, replay behavior, fake motion patterns.
Tech Stack: MediaPipe Tasks FaceLandmarker (landmarks) + OpenCV + NumPy
Output: L1 liveness score + motion anomaly reason + status
"""
import cv2
import numpy as np
from collections import deque
from scipy.spatial import distance as dist
from typing import Optional

from app.core.types import FrameData
from app.core import config
from app.modules.face_detector import FaceDetector


class TemporalLivenessEngine:
    """
    L1: Temporal Liveness Integrity
    Detects replay attacks, frozen frames, unnatural head pose stability,
    blink patterns over time.

    Frozen-frame detection (perceptual hashing) works without MediaPipe.
    Landmark-based checks (EAR, head pose, blink variation) require the
    MediaPipe Tasks FaceLandmarker; when unavailable the layer reports
    UNAVAILABLE and never fabricates a liveness prediction.
    """

    def __init__(self, history_len: int = 30, face_detector: Optional[FaceDetector] = None):
        self.face_detector = face_detector
        self.use_mediapipe = bool(face_detector is not None and face_detector.use_mediapipe)

        # Rolling history buffers
        self.yaw_history = deque(maxlen=history_len)
        self.pitch_history = deque(maxlen=history_len)
        self.ear_history = deque(maxlen=history_len)
        self.frame_hashes = deque(maxlen=history_len)
        self.prev_gray: Optional[np.ndarray] = None
        self.static_frame_count: int = 0

        # Landmark indices (MediaPipe FaceMesh / FaceLandmarker numbering)
        self.LEFT_EYE = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE = [362, 385, 387, 263, 373, 380]

        self.status = (
            config.LAYER_STATUS_UNAVAILABLE
            if not self.use_mediapipe
            else config.LAYER_STATUS_PASS
        )
        if not self.use_mediapipe:
            print("L1-TemporalLiveness: MediaPipe FaceLandmarker unavailable — "
                  "reporting UNAVAILABLE (frozen-frame check only)")

    # ── Public API ────────────────────────────────────────────────────────
    def process(self, frame_data: FrameData) -> None:
        frame = frame_data.image
        if frame is None or frame.size == 0:
            frame_data.liveness_score = None
            frame_data.flags.append("L1_NO_FRAME")
            frame_data.metadata["l1_status"] = config.LAYER_STATUS_ERROR
            self.status = config.LAYER_STATUS_ERROR
            return

        h, w = frame.shape[:2]

        # ── Frozen-frame / replay detection (pixel sensor noise check) ──
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frozen = self._check_frozen(gray)
        frame_data.metadata["l1_frozen"] = frozen

        # ── Landmark-based pose + blink analysis (MediaPipe) ──
        if not self.use_mediapipe or self.face_detector is None:
            frame_data.liveness_score = None
            frame_data.metadata["l1_status"] = config.LAYER_STATUS_UNAVAILABLE
            self.status = config.LAYER_STATUS_UNAVAILABLE
            return

        landmarks = self.face_detector.landmarks(frame)
        if landmarks is None:
            frame_data.liveness_score = None
            frame_data.flags.append("L1_NO_FACE_MESH")
            frame_data.metadata["l1_status"] = config.LAYER_STATUS_INCONCLUSIVE
            self.status = config.LAYER_STATUS_INCONCLUSIVE
            return

        def pt(idx):
            return (landmarks[idx][0] * w, landmarks[idx][1] * h)

        ear = self._ear(landmarks)
        self.ear_history.append(ear)
        yaw, pitch = self._head_pose(landmarks, w, h)
        self.yaw_history.append(yaw)
        self.pitch_history.append(pitch)

        score = 1.0
        reasons = []

        if len(self.yaw_history) >= 10:
            yaw_var = float(np.var(list(self.yaw_history)))
            pitch_var = float(np.var(list(self.pitch_history)))
            if yaw_var < 0.35 and pitch_var < 0.35:
                score -= 0.50
                reasons.append("L1_UNNATURAL_HEAD_STABILITY")

        if len(self.ear_history) >= 12:
            ear_var = float(np.var(list(self.ear_history)))
            if ear_var < 0.0001:
                score -= 0.60
                reasons.append("L1_NO_BLINK_VARIATION")

        # In a real live human, total absence of blink AND total absence of micro-movement AND frozen pixels
        if "L1_NO_BLINK_VARIATION" in reasons and "L1_UNNATURAL_HEAD_STABILITY" in reasons:
            if frame_data.metadata.get("l1_frozen", False):
                score = min(score, 0.05)
                reasons.append("L1_STATIC_PHOTO_DETECTED")
            else:
                score -= 0.20

        if abs(yaw) > 40 or abs(pitch) > 35:
            score -= 0.2
            reasons.append("L1_EXTREME_HEAD_POSE")

        score = float(np.clip(score, 0.0, 1.0))
        frame_data.liveness_score = score
        frame_data.flags.extend(reasons)
        frame_data.metadata.update({
            "landmarks": landmarks,
            "l1_yaw": round(yaw, 2),
            "l1_pitch": round(pitch, 2),
            "l1_ear": round(ear, 4),
            "l1_status": config.LAYER_STATUS_PASS if not reasons else config.LAYER_STATUS_FAIL,
        })
        self.status = config.LAYER_STATUS_PASS

    # ── Helpers ───────────────────────────────────────────────────────────
    def _check_frozen(self, gray: np.ndarray) -> bool:
        if self.prev_gray is not None and gray.shape == self.prev_gray.shape:
            # Camera sensors always have slight electronic noise; true frozen digital frame has ~0 diff
            diff = float(np.mean(cv2.absdiff(gray, self.prev_gray)))
            if diff < 0.20:
                self.static_frame_count += 1
            else:
                self.static_frame_count = max(0, self.static_frame_count - 1)
        else:
            self.static_frame_count = 0
        self.prev_gray = gray.copy()
        return self.static_frame_count >= 15

    def _ear(self, landmarks) -> float:
        def coords(indices):
            return [(landmarks[i][0], landmarks[i][1]) for i in indices]

        le = coords(self.LEFT_EYE)
        re = coords(self.RIGHT_EYE)

        def _e(eye):
            A = dist.euclidean(eye[1], eye[5])
            B = dist.euclidean(eye[2], eye[4])
            C = dist.euclidean(eye[0], eye[3])
            return (A + B) / (2.0 * C + 1e-6)

        return float((_e(le) + _e(re)) / 2.0)

    def _head_pose(self, landmarks, w, h):
        # Approximate PnP head pose using normalized landmark positions.
        model_pts = np.array([
            (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
            (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
            (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
        ])
        img_pts = np.array([
            (landmarks[1][0] * w, landmarks[1][1] * h),      # nose
            (landmarks[152][0] * w, landmarks[152][1] * h),  # chin
            (landmarks[33][0] * w, landmarks[33][1] * h),    # left eye
            (landmarks[263][0] * w, landmarks[263][1] * h),  # right eye
            (landmarks[61][0] * w, landmarks[61][1] * h),    # mouth left
            (landmarks[291][0] * w, landmarks[291][1] * h),  # mouth right
        ], dtype="double")
        fl = w
        cam = np.array([[fl, 0, w / 2], [0, fl, h / 2], [0, 0, 1]], dtype="double")
        dc = np.zeros((4, 1))
        ok, rvec, _ = cv2.solvePnP(model_pts, img_pts, cam, dc)
        if not ok:
            return 0.0, 0.0
        rmat, _ = cv2.Rodrigues(rvec)
        angles, *_ = cv2.RQDecomp3x3(rmat)
        pitch = float(angles[0])
        yaw = float(angles[1])
        if abs(pitch) > 90:
            pitch = (180.0 - abs(pitch)) * (-1.0 if pitch > 0 else 1.0)
        if abs(yaw) > 90:
            yaw = (180.0 - abs(yaw)) * (-1.0 if yaw > 0 else 1.0)
        return float(yaw), float(pitch)  # yaw, pitch normalized to [-90, +90]
