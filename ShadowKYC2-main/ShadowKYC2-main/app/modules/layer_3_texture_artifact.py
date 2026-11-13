"""
Layer 3 – Deep Neural Anti-Spoofing & Texture/Frequency Artifact Detection
Purpose: Detect photo replacement (printed photos), screen replays, AI generated videos, and GAN artifacts.
Tech Stack: MiniFASNet V2 (ONNX Runtime) + FFT (NumPy) + Laplacian Variance
Output: L3 anti-spoofing score + artifact reasons + status
"""
import cv2
import numpy as np
import onnxruntime as ort
from pathlib import Path

from app.core.types import FrameData
from app.core import config


class TextureArtifactDetector:
    """
    L3: Deep Neural Anti-Spoofing + Frequency / Texture Analysis
    
    Combines:
    1. MiniFASNet V2 Deep Convolutional Network:
       Multi-scale presentation attack detector trained on real vs print vs screen replay.
       Outputs 3-class probabilities:
         - Class 0: Print Attack (printed paper photo, cutouts)
         - Class 1: Real Live 3D Human Face
         - Class 2: Screen Replay Attack (phone/monitor screen, AI video playback, deepfakes)
    2. 2D Fast Fourier Transform (FFT) High-Frequency Energy Ratio:
       Catches GAN frequency grid anomalies and diffusion smoothing.
    3. Laplacian & Local Binary Pattern (LBP) Texture Variance:
       Identifies unnatural synthetic skin smoothing and planar reflection patterns.
    """

    BLUR_THRESHOLD     = 70.0    # Below -> suspiciously smooth (synthetic/diffused face)
    FFT_LOW_THRESHOLD  = 0.06    # Below -> missing high-freq biological details
    FFT_HIGH_THRESHOLD = 0.48    # Above -> compression / GAN grid artifact
    LBP_LOW_THRESHOLD  = 18.0    # Below -> unnaturally uniform synthetic skin texture

    def __init__(self):
        self.minifasnet_session = None
        model_path = getattr(config, "MINIFASNET_MODEL_PATH", config.MODELS_DIR / "minifasnet_v2.onnx")
        if Path(model_path).exists():
            try:
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = 2
                self.minifasnet_session = ort.InferenceSession(str(model_path), opts)
                print(f"L3-AntiSpoof: Loaded MiniFASNet V2 deep neural classifier ({model_path.name})")
            except Exception as e:
                print(f"L3-AntiSpoof: Failed to load MiniFASNet ONNX ({e})")
        else:
            print(f"L3-AntiSpoof: Warning - MiniFASNet model not found at {model_path}")

    # ── Public API ────────────────────────────────────────────────────────────
    def process(self, frame_data: FrameData) -> None:
        face = frame_data.face_roi if frame_data.face_roi is not None else frame_data.image
        if face is None or face.size == 0:
            frame_data.artifact_score = 0.10
            frame_data.flags.append("L3_NO_FACE_TEXTURE")
            frame_data.metadata["l3_status"] = config.LAYER_STATUS_FAIL
            return

        score = 1.0
        reasons = []

        # ── 1. MiniFASNet Deep Neural Anti-Spoofing Inference ───────────────
        p_print = 0.0
        p_live = 1.0
        p_replay = 0.0

        if self.minifasnet_session is not None:
            try:
                crop_80 = cv2.resize(face, (80, 80))
                inp = crop_80.transpose(2, 0, 1).astype(np.float32) / 255.0
                inp = np.expand_dims(inp, axis=0)

                out = self.minifasnet_session.run(None, {"input": inp})[0][0]
                exp_out = np.exp(out - np.max(out))
                probs = exp_out / np.sum(exp_out)

                # MiniFASNet V2 Class Indexing:
                # Class 0: Print photo attack
                # Class 1: Screen replay attack
                # Class 2: Genuine live 3D human face
                p_print = float(probs[0])
                p_replay = float(probs[1])
                p_live = float(probs[2])

                frame_data.metadata["l3_p_print"] = round(p_print, 4)
                frame_data.metadata["l3_p_live"] = round(p_live, 4)
                frame_data.metadata["l3_p_replay"] = round(p_replay, 4)

                # Calibrated Spoof Detection: screen replays & print photo
                # Distinguish true presentation attacks from ordinary webcam compression
                is_active_live = frame_data.metadata.get("active_challenge_completed", False)

                if p_live > 0.60:
                    # Confirmed genuine live human face by deep neural classifier
                    score = max(score, min(0.96, p_live))
                elif p_replay > 0.80 and not is_active_live:
                    score = min(score, 0.05)
                    reasons.append("L3_SCREEN_REPLAY_DETECTED")
                elif p_print > 0.75 and not is_active_live:
                    score = min(score, 0.05)
                    reasons.append("L3_PRINT_PHOTO_DETECTED")
                elif p_live < 0.25 and not is_active_live:
                    score = min(score, 0.20)
                    reasons.append("L3_LOW_LIVE_PROBABILITY")

            except Exception as e:
                print(f"[L3] MiniFASNet inference error: {e}")

        # ── 2. Frequency & Biological Texture Checks ───────────────────────
        gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) if face.ndim == 3 else face.copy()
        gray = cv2.resize(gray, (128, 128))

        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if lap_var < 30.0 and p_live < 0.60:
            score -= 0.25
            reasons.append("L3_OVER_SMOOTH_FACE")

        fft_energy_ratio = self._fft_hf_ratio(gray)
        if fft_energy_ratio < 0.035 and p_live < 0.60:
            score -= 0.25
            reasons.append("L3_LOW_HF_ENERGY_AI_FACE")
        elif fft_energy_ratio > 0.52:
            score -= 0.25
            reasons.append("L3_GAN_GRID_ARTIFACT")

        lbp_var = self._lbp_variance(gray)
        if lbp_var < 10.0 and p_live < 0.60:
            score -= 0.20
            reasons.append("L3_UNIFORM_SKIN_TEXTURE")

        final_score = float(np.clip(score, 0.0, 1.0))
        frame_data.artifact_score = final_score
        frame_data.flags.extend(reasons)
        frame_data.metadata.update({
            "l3_laplacian_var": round(lap_var, 2),
            "l3_fft_hf_ratio": round(fft_energy_ratio, 4),
            "l3_lbp_var": round(lbp_var, 2),
            "l3_status": config.LAYER_STATUS_PASS if not reasons else config.LAYER_STATUS_FAIL,
        })

    # ── Helpers ───────────────────────────────────────────────────────────────
    @staticmethod
    def _fft_hf_ratio(gray: np.ndarray) -> float:
        f = np.fft.fft2(gray.astype(np.float32))
        fshift = np.fft.fftshift(f)
        magnitude = np.abs(fshift)

        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        radius = min(h, w) // 4

        Y, X = np.ogrid[:h, :w]
        mask_low = (X - cx)**2 + (Y - cy)**2 <= radius**2

        total_energy = float(magnitude.sum()) + 1e-8
        hf_energy    = float(magnitude[~mask_low].sum())
        return hf_energy / total_energy

    @staticmethod
    def _lbp_variance(gray: np.ndarray) -> float:
        padded = np.pad(gray.astype(np.float32), 1, mode='edge')
        center = padded[1:-1, 1:-1]
        lbp = np.zeros_like(center, dtype=np.uint8)
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue
                neighbor = padded[1+dy:padded.shape[0]-1+dy, 1+dx:padded.shape[1]-1+dx]
                lbp = (lbp << 1) | (neighbor >= center).astype(np.uint8)
        return float(np.var(lbp))
