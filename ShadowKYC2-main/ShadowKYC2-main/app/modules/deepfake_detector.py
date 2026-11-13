"""
Deepfake & Synthetic Face Forensic Detection Engine
Detects:
1. Face-Swap Blending Seams & Boundary Color Discontinuity
2. Corneal Specular Reflection Asymmetry & Pupil Light Mismatch
3. High-Frequency Fourier Spectral Grid Anomalies (GAN transposed convolutions)
4. Temporal Edge Flutter & Jitter (Frame-to-Frame Blending Mask Instability)
5. Over-smoothed Diffusion Artifacts & Plastic Skin Surfaces
"""
import cv2
import numpy as np
from typing import Dict, List, Optional, Tuple
from collections import deque

from app.core.types import FrameData
from app.core import config


class DeepfakeDetector:
    """
    Multi-Feature Forensic Deepfake Detector
    Runs real-time multi-modal forensic checks on face ROIs and landmarks.
    """

    SEAM_GRADIENT_THRESHOLD = 48.0      # Boundary edge gradient jump
    COLOR_DELTA_THRESHOLD   = 45.0      # Delta E chromaticity gap between face & neck
    CORNEAL_MISMATCH_THRESH = 0.45      # Specular reflection discrepancy
    FFT_GRID_THRESHOLD      = 0.46      # High-frequency periodic GAN artifact
    TEMPORAL_JITTER_THRESH  = 40.0      # Frame-to-frame mask boundary flicker

    def __init__(self, history_len: int = 15):
        self.history_len = history_len
        self.boundary_history: deque = deque(maxlen=history_len)
        self.prev_face_gray: Optional[np.ndarray] = None
        self.frame_count = 0

    def process(self, frame_data: FrameData) -> Dict:
        """
        Analyze the frame for deepfake, face-swap, and synthetic AI generation artifacts.
        Updates frame_data.metadata and returns deepfake telemetry.
        """
        frame = frame_data.image
        face = frame_data.face_roi if frame_data.face_roi is not None else frame
        landmarks = frame_data.metadata.get("landmarks")

        if face is None or face.size == 0:
            frame_data.metadata["deepfake_probability"] = 0.0
            frame_data.metadata["deepfake_score"] = 0.50
            return {"probability": 0.0, "is_deepfake": False, "flags": []}

        self.frame_count += 1
        flags = []
        p_deepfake = 0.0

        h, w = face.shape[:2]
        face_gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.ndim == 3 else face.copy()
        face_gray_norm = cv2.resize(face_gray, (160, 160))

        # ── 1. Facial Boundary Seam & Chromaticity Discontinuity ─────────
        seam_score, color_delta = self._check_boundary_seam(frame, face)
        if seam_score > self.SEAM_GRADIENT_THRESHOLD and color_delta > self.COLOR_DELTA_THRESHOLD:
            flags.append("DEEPFAKE_BLENDING_SEAM_DETECTED")
            p_deepfake += 0.45

        # ── 2. Corneal Specular Reflection Symmetry ─────────────────────
        if landmarks is not None and len(landmarks) > 386:
            corneal_diff = self._check_corneal_reflection(frame, landmarks)
            if corneal_diff > self.CORNEAL_MISMATCH_THRESH:
                flags.append("DEEPFAKE_CORNEAL_SPECULAR_MISMATCH")
                p_deepfake += 0.30

        # ── 3. FFT Frequency Grid / GAN Checkerboard Anomaly ─────────────
        fft_ratio = self._fft_anomaly(face_gray_norm)
        if fft_ratio > self.FFT_GRID_THRESHOLD:
            flags.append("DEEPFAKE_GAN_GRID_ARTIFACT")
            p_deepfake += 0.35
        elif fft_ratio < 0.045:
            flags.append("DEEPFAKE_SYNTHETIC_DIFFUSION_SMOOTH")
            p_deepfake += 0.25

        # ── 4. Temporal Mask Edge Flutter / Frame-to-Frame Jitter ────────
        if self.prev_face_gray is not None:
            diff = cv2.absdiff(face_gray_norm, self.prev_face_gray)
            # Mask edges: outer 15% perimeter
            mask = np.zeros((160, 160), dtype=np.uint8)
            mask[:24, :] = 1
            mask[-24:, :] = 1
            mask[:, :24] = 1
            mask[:, -24:] = 1
            perimeter_jitter = float(np.mean(diff[mask == 1]))
            if perimeter_jitter > self.TEMPORAL_JITTER_THRESH:
                flags.append("DEEPFAKE_TEMPORAL_MASK_FLUTTER")
                p_deepfake += 0.25
        self.prev_face_gray = face_gray_norm.copy()

        # MiniFASNet neural replay cross-check
        p_replay = frame_data.metadata.get("l3_p_replay", 0.0)
        p_live = frame_data.metadata.get("l3_p_live", 0.5)
        if p_replay > 0.82 and p_live < 0.20 and not frame_data.metadata.get("active_challenge_completed", False):
            flags.append("DEEPFAKE_SCREEN_SYNTHESIS")
            p_deepfake += 0.40
        elif p_live > 0.70:
            # Genuine biological face confirmed by neural classifier
            p_deepfake *= 0.20

        # If active challenge is verified on real physical 3D movement, dampen false noise
        if frame_data.metadata.get("active_challenge_completed", False):
            p_deepfake = min(p_deepfake * 0.35, 0.25)
            # Remove speculative flags if physical 3D interaction was proven
            flags = [f for f in flags if f in ("DEEPFAKE_BLENDING_SEAM_DETECTED", "DEEPFAKE_GAN_GRID_ARTIFACT")]

        final_prob = float(np.clip(p_deepfake, 0.0, 1.0))
        is_deepfake = final_prob >= 0.65

        if is_deepfake and "DEEPFAKE_SYNTHETIC_FACE_DETECTED" not in flags:
            flags.insert(0, "DEEPFAKE_SYNTHETIC_FACE_DETECTED")

        # Update frame metadata
        frame_data.metadata.update({
            "deepfake_probability": round(final_prob, 4),
            "deepfake_score": round(1.0 - final_prob, 4),
            "deepfake_flags": flags,
            "deepfake_detected": is_deepfake,
        })
        frame_data.flags.extend(flags)

        return {
            "probability": round(final_prob, 4),
            "score": round(1.0 - final_prob, 4),
            "is_deepfake": is_deepfake,
            "flags": flags,
        }

    # ── Forensic Sub-routines ────────────────────────────────────────────────
    @staticmethod
    def _check_boundary_seam(frame: np.ndarray, face: np.ndarray) -> Tuple[float, float]:
        """Detect edge seam gradient and CIELAB color shift between face and perimeter border."""
        try:
            if face.shape[0] < 30 or face.shape[1] < 30:
                return 0.0, 0.0
            
            h, w = face.shape[:2]
            lab = cv2.cvtColor(face, cv2.COLOR_BGR2LAB)
            inner_lab = lab[int(h * 0.25):int(h * 0.75), int(w * 0.25):int(w * 0.75)]
            border_lab = np.concatenate([
                lab[:int(h * 0.15), :].reshape(-1, 3),
                lab[-int(h * 0.15):, :].reshape(-1, 3),
                lab[:, :int(w * 0.15)].reshape(-1, 3),
                lab[:, -int(w * 0.15):].reshape(-1, 3),
            ])

            mean_inner = np.mean(inner_lab, axis=(0, 1))
            mean_border = np.mean(border_lab, axis=0)
            color_delta = float(np.linalg.norm(mean_inner - mean_border))

            # Sobel gradient across transition margin (10% to 30% border zone where blending masks cut off)
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
            
            ring_mask = np.zeros((h, w), dtype=bool)
            ring_mask[int(h * 0.1):int(h * 0.3), :] = True
            ring_mask[int(h * 0.7):int(h * 0.9), :] = True
            ring_mask[:, int(w * 0.1):int(w * 0.3)] = True
            ring_mask[:, int(w * 0.7):int(w * 0.9)] = True
            edge_seam = float(np.mean(magnitude[ring_mask]))

            return edge_seam, color_delta
        except Exception:
            return 0.0, 0.0

    @staticmethod
    def _check_corneal_reflection(frame: np.ndarray, landmarks) -> float:
        """Compare specular reflection points in left pupil vs right pupil."""
        try:
            h, w = frame.shape[:2]
            # Left eye pupil landmark ~468 (or center of 33, 133), Right ~473 (or center 362, 263)
            p_left = (int(landmarks[33][0] * w), int(landmarks[33][1] * h))
            p_right = (int(landmarks[263][0] * w), int(landmarks[263][1] * h))

            r = 12
            left_patch = frame[max(0, p_left[1]-r):min(h, p_left[1]+r), max(0, p_left[0]-r):min(w, p_left[0]+r)]
            right_patch = frame[max(0, p_right[1]-r):min(h, p_right[1]+r), max(0, p_right[0]-r):min(w, p_right[0]+r)]

            if left_patch.size == 0 or right_patch.size == 0:
                return 0.0

            left_gray = cv2.cvtColor(left_patch, cv2.COLOR_BGR2GRAY)
            right_gray = cv2.cvtColor(right_patch, cv2.COLOR_BGR2GRAY)

            # Brightest reflection hotspot
            l_max = float(np.max(left_gray))
            r_max = float(np.max(right_gray))
            diff = abs(l_max - r_max) / (max(l_max, r_max) + 1e-6)
            return float(diff)
        except Exception:
            return 0.0

    @staticmethod
    def _fft_anomaly(gray: np.ndarray) -> float:
        """High frequency Fourier energy distribution for GAN grid identification."""
        try:
            f = np.fft.fft2(gray.astype(np.float32))
            fshift = np.fft.fftshift(f)
            magnitude = np.abs(fshift)
            h, w = magnitude.shape
            cy, cx = h // 2, w // 2
            radius = min(h, w) // 4
            Y, X = np.ogrid[:h, :w]
            mask_low = (X - cx)**2 + (Y - cy)**2 <= radius**2
            total = float(magnitude.sum()) + 1e-8
            hf = float(magnitude[~mask_low].sum())
            return hf / total
        except Exception:
            return 0.10
