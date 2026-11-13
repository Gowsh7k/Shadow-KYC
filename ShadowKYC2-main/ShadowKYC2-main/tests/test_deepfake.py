import numpy as np
import pytest
from app.core.types import FrameData
from app.modules.deepfake_detector import DeepfakeDetector
from app.core.orchestrator import SessionOrchestrator


def test_deepfake_detector_clean_face():
    detector = DeepfakeDetector()
    # Generate simulated natural skin face ROI (160x160x3)
    np.random.seed(42)
    natural_face = np.full((160, 160, 3), 140, dtype=np.uint8)
    noise = np.random.randint(-10, 10, natural_face.shape, dtype=np.int16)
    natural_face = np.clip(natural_face.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    frame_data = FrameData(image=natural_face, face_roi=natural_face, timestamp_sec=0.0)
    frame_data.metadata["l3_p_live"] = 0.90
    res = detector.process(frame_data)

    assert "probability" in res
    assert "is_deepfake" in res
    assert res["is_deepfake"] is False
    assert res["probability"] < 0.65
    assert frame_data.metadata.get("deepfake_detected") is False


def test_deepfake_detector_face_swap_boundary_seam():
    detector = DeepfakeDetector()
    # Generate face ROI with an artificial boundary mismatch (face swap seam)
    face = np.zeros((160, 160, 3), dtype=np.uint8)
    # Bright center (swapped face)
    face[25:135, 25:135] = [255, 255, 255]
    # Dark perimeter (original background/neck)
    face[:25, :] = [0, 0, 0]
    face[135:, :] = [0, 0, 0]
    face[:, :25] = [0, 0, 0]
    face[:, 135:] = [0, 0, 0]

    frame_data = FrameData(image=face, face_roi=face, timestamp_sec=0.0)
    frame_data.metadata["l3_p_live"] = 0.10
    res = detector.process(frame_data)

    assert "DEEPFAKE_BLENDING_SEAM_DETECTED" in res["flags"]
    assert res["probability"] > 0.40


def test_deepfake_detector_gan_checkerboard():
    detector = DeepfakeDetector()
    # Generate high frequency checkerboard pattern (characteristic of transposed convolution)
    checkerboard = np.indices((160, 160)).sum(axis=0) % 2 * 255
    checkerboard_3ch = np.stack([checkerboard]*3, axis=-1).astype(np.uint8)

    frame_data = FrameData(image=checkerboard_3ch, face_roi=checkerboard_3ch, timestamp_sec=0.0)
    frame_data.metadata["l3_p_live"] = 0.10
    res = detector.process(frame_data)

    assert "DEEPFAKE_GAN_GRID_ARTIFACT" in res["flags"]
    assert res["probability"] > 0.30


def test_orchestrator_integrates_deepfake_engine():
    orch = SessionOrchestrator()
    assert hasattr(orch, "deepfake_detector")
    assert isinstance(orch.deepfake_detector, DeepfakeDetector)

    # Process a clean dummy frame
    frame = np.full((480, 640, 3), 120, dtype=np.uint8)
    data = orch.process_single_frame(frame, timestamp=0.0)

    assert "deepfake_probability" in data
    assert "deepfake_detected" in data
    assert "deepfake_flags" in data
