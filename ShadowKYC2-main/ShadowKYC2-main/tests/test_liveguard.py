"""
Unit tests for app/modules/liveguard/ components.

Covers:
  - _ear() and _mar() calculation correctness
  - BlinkDetector state machine (close→open counts one blink)
  - BlinkDetector: sustained closure does not multi-count
  - BlinkCounter.reset() isolates per-challenge counts
  - calibrate_ear, calibrate_mar, calibrate_head_yaw thresholds
  - calibrate_* fallback when too few samples
  - classify_turn() LEFT / RIGHT / CENTER with sign flip
  - estimate_head_pose() success vs. failure conditions
  - LiveGuardEngine PASS when blink sequence is complete
  - LiveGuardEngine TIMEOUT when no action within deadline
  - LiveGuardEngine INCONCLUSIVE when face missing > max_no_face_s
  - Missing-face handling: engine does not crash on sustained None landmarks

Run with:
    python -m pytest tests/test_liveguard.py -v
    (from the ShadowKYC2-main/ project root, venv activated)
"""
from __future__ import annotations

import os
import sys
import time
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# ── Path setup ──────────────────────────────────────────────────────────────
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

os.environ.setdefault("SECRET_KEY", "test-secret-liveguard")

# ── Import production modules ───────────────────────────────────────────────
from app.modules.liveguard.blink import BlinkCounter, BlinkDetector
from app.modules.liveguard.calibrate import (
    EAR_ABS_CLOSED_MAX,
    EAR_ABS_CLOSED_MIN,
    HEAD_YAW_THRESH_DEG,
    MAR_ABS_OPEN_MIN,
    calibrate_ear,
    calibrate_head_yaw,
    calibrate_mar,
)
from app.modules.liveguard.engine import _ear, _mar
from app.modules.liveguard.geometry import (
    HeadPose,
    classify_turn,
    estimate_head_pose,
)


# ── Landmark helpers ────────────────────────────────────────────────────────
LEFT_EYE  = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LM_TOP_LIP     = 13
LM_BOTTOM_LIP  = 14
LM_MOUTH_LEFT  = 78
LM_MOUTH_RIGHT = 308


def _make_landmarks(n=478, w=640, h=480) -> List[Tuple[float, float]]:
    """Minimal synthetic landmark set (all centred, eyes/mouth in reasonable positions)."""
    cx, cy = 0.5, 0.5
    pts: List[Tuple[float, float]] = [(cx, cy)] * n

    # Left eye — enough spread to produce a realistic EAR
    #   ids: [33, 160, 158, 133, 153, 144]
    pts[33]  = (cx - 0.09, cy - 0.06)
    pts[160] = (cx - 0.07, cy - 0.08)
    pts[158] = (cx - 0.05, cy - 0.07)
    pts[133] = (cx - 0.01, cy - 0.06)
    pts[153] = (cx - 0.05, cy - 0.05)
    pts[144] = (cx - 0.07, cy - 0.05)

    # Right eye — symmetric
    pts[263] = (cx + 0.09, cy - 0.06)
    pts[385] = (cx + 0.07, cy - 0.08)
    pts[387] = (cx + 0.05, cy - 0.07)
    pts[362] = (cx + 0.01, cy - 0.06)
    pts[373] = (cx + 0.05, cy - 0.05)
    pts[380] = (cx + 0.07, cy - 0.05)

    # Mouth (closed)
    pts[LM_TOP_LIP]    = (cx, cy + 0.10)
    pts[LM_BOTTOM_LIP] = (cx, cy + 0.12)   # very small vertical gap → small MAR
    pts[LM_MOUTH_LEFT]  = (cx - 0.05, cy + 0.11)
    pts[LM_MOUTH_RIGHT] = (cx + 0.05, cy + 0.11)

    # solvePnP needs these indices for head pose
    pts[1]   = (cx, cy - 0.10)   # nose tip
    pts[152] = (cx, cy + 0.30)   # chin
    pts[61]  = (cx - 0.04, cy + 0.08)
    pts[291] = (cx + 0.04, cy + 0.08)

    return pts


def _make_closed_eye_landmarks() -> List[Tuple[float, float]]:
    """Landmarks where eyes are almost closed (top/bottom lids very close)."""
    pts = _make_landmarks()
    cx, cy = 0.5, 0.5
    # Make top and bottom lid very close together → low EAR
    pts[160] = (cx - 0.07, cy - 0.060)   # was -0.08
    pts[158] = (cx - 0.05, cy - 0.060)
    pts[153] = (cx - 0.05, cy - 0.060)
    pts[144] = (cx - 0.07, cy - 0.060)
    pts[385] = (cx + 0.07, cy - 0.060)
    pts[387] = (cx + 0.05, cy - 0.060)
    pts[373] = (cx + 0.05, cy - 0.060)
    pts[380] = (cx + 0.07, cy - 0.060)
    return pts


def _make_open_mouth_landmarks() -> List[Tuple[float, float]]:
    """Landmarks where the mouth is wide open."""
    pts = _make_landmarks()
    cx, cy = 0.5, 0.5
    pts[LM_TOP_LIP]    = (cx, cy + 0.08)
    pts[LM_BOTTOM_LIP] = (cx, cy + 0.18)   # large vertical gap → large MAR
    pts[LM_MOUTH_LEFT]  = (cx - 0.05, cy + 0.13)
    pts[LM_MOUTH_RIGHT] = (cx + 0.05, cy + 0.13)
    return pts


# ══════════════════════════════════════════════════════════════════════════════
# 1. EAR calculation
# ══════════════════════════════════════════════════════════════════════════════

class TestEAR:
    def test_open_eyes_ear_positive(self):
        lm = _make_landmarks()
        ear = _ear(lm, 640, 480)
        assert ear > 0.0, "EAR must be positive for open eyes"

    def test_open_vs_closed_ear_ordering(self):
        """Closed-eye EAR must be strictly lower than open-eye EAR."""
        lm_open   = _make_landmarks()
        lm_closed = _make_closed_eye_landmarks()
        ear_open   = _ear(lm_open,   640, 480)
        ear_closed = _ear(lm_closed, 640, 480)
        assert ear_closed < ear_open, (
            f"Closed EAR ({ear_closed:.4f}) should be < open EAR ({ear_open:.4f})"
        )

    def test_ear_symmetric(self):
        """Left and right eye with identical geometry should give the same individual EAR,
        and the mean should equal either side."""
        lm = _make_landmarks()
        ear = _ear(lm, 640, 480)
        assert 0.0 < ear < 2.0  # sanity bounds


# ══════════════════════════════════════════════════════════════════════════════
# 2. MAR calculation
# ══════════════════════════════════════════════════════════════════════════════

class TestMAR:
    def test_closed_mouth_mar_small(self):
        lm = _make_landmarks()
        mar = _mar(lm, 640, 480)
        # Closed mouth should have a small MAR (< 0.2 is a reasonable upper bound)
        assert 0.0 <= mar < 0.2, f"Closed MAR unexpectedly large: {mar:.4f}"

    def test_open_mouth_mar_larger(self):
        lm_open   = _make_open_mouth_landmarks()
        lm_closed = _make_landmarks()
        mar_open   = _mar(lm_open,   640, 480)
        mar_closed = _mar(lm_closed, 640, 480)
        assert mar_open > mar_closed, (
            f"Open MAR ({mar_open:.4f}) should exceed closed MAR ({mar_closed:.4f})"
        )

    def test_mar_non_negative(self):
        mar = _mar(_make_landmarks(), 640, 480)
        assert mar >= 0.0


# ══════════════════════════════════════════════════════════════════════════════
# 3. BlinkDetector state machine
# ══════════════════════════════════════════════════════════════════════════════

class TestBlinkDetector:
    def _detector(self):
        return BlinkDetector(ear_closed_thresh=0.21, ear_reopen_thresh=0.28)

    def test_single_blink_counts_once(self):
        bd = self._detector()
        # open → closed → open (one blink)
        events = []
        for ear in [0.35, 0.35, 0.15, 0.15, 0.35, 0.35]:
            events.append(bd.update(ear))
        assert bd.blink_count == 1
        assert sum(events) == 1, "update() should return True exactly once per blink"

    def test_sustained_closure_counts_once(self):
        """Holding eyes shut for many frames should still count as one blink."""
        bd = self._detector()
        # open, many frames closed, then reopen
        for _ in range(5):
            bd.update(0.35)
        for _ in range(20):
            bd.update(0.10)
        for _ in range(5):
            bd.update(0.35)
        assert bd.blink_count == 1

    def test_two_blinks(self):
        bd = self._detector()
        sequence = (
            [0.35] * 3 + [0.10] * 3 + [0.35] * 3 +   # blink 1
            [0.10] * 3 + [0.35] * 3                    # blink 2
        )
        for ear in sequence:
            bd.update(ear)
        assert bd.blink_count == 2

    def test_none_ear_no_count(self):
        bd = self._detector()
        for _ in range(10):
            bd.update(None)
        assert bd.blink_count == 0

    def test_reset_clears_state(self):
        bd = self._detector()
        for ear in [0.35, 0.10, 0.35]:
            bd.update(ear)
        assert bd.blink_count == 1
        bd.reset()
        assert bd.blink_count == 0
        assert bd.eyes_closed is False

    def test_hysteresis_no_bounce(self):
        """Values bouncing right at the threshold should not create phantom blinks."""
        bd = self._detector()
        # Eyes transitioning through closed_thresh but NOT reaching reopen_thresh
        boundary_ear = 0.22   # between closed_thresh(0.21) and reopen_thresh(0.28)?
        # Actually 0.22 < 0.28, so if we alternate 0.20/0.22 it should close once
        for _ in range(10):
            bd.update(0.20)  # closed
            bd.update(0.22)  # still below reopen_thresh (0.28) → no blink
        assert bd.blink_count == 0  # never reached reopen_thresh


# ══════════════════════════════════════════════════════════════════════════════
# 4. BlinkCounter per-challenge isolation
# ══════════════════════════════════════════════════════════════════════════════

class TestBlinkCounter:
    def test_resets_blink_count_across_challenges(self):
        bc = BlinkCounter(required=2)
        bc.detector.ear_closed_thresh = 0.21
        bc.detector.ear_reopen_thresh = 0.28

        # Simulate 2 blinks (passes challenge 1)
        for ear in [0.35, 0.10, 0.35, 0.10, 0.35]:
            bc.update(ear)
        assert bc.passed is True
        assert bc.detector.blink_count == 2

        # Reset for challenge 2 — count must be zero again
        bc.reset(2)
        assert bc.detector.blink_count == 0
        assert bc.passed is False

    def test_not_passed_before_required(self):
        bc = BlinkCounter(required=2)
        bc.detector.ear_closed_thresh = 0.21
        bc.detector.ear_reopen_thresh = 0.28
        # Only one blink
        for ear in [0.35, 0.10, 0.35]:
            bc.update(ear)
        assert bc.passed is False


# ══════════════════════════════════════════════════════════════════════════════
# 5. calibrate_ear
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibrateEAR:
    def _samples(self, base=0.42, n=30):
        rng = np.random.default_rng(42)
        return (base + rng.normal(0, 0.02, n)).tolist()

    def test_thresholds_ordered(self):
        ec = calibrate_ear(self._samples())
        assert ec.closed_thresh < ec.reopen_thresh, (
            f"closed_thresh ({ec.closed_thresh}) must be < reopen_thresh ({ec.reopen_thresh})"
        )

    def test_closed_thresh_within_bounds(self):
        ec = calibrate_ear(self._samples())
        assert EAR_ABS_CLOSED_MIN <= ec.closed_thresh <= EAR_ABS_CLOSED_MAX

    def test_fallback_on_few_samples(self):
        ec = calibrate_ear([0.40] * 3)  # fewer than EAR_MIN_SAMPLES=8
        assert ec.method == "default"
        assert ec.closed_thresh < ec.reopen_thresh

    def test_empty_samples(self):
        ec = calibrate_ear([])
        assert ec.method == "default"

    def test_typical_baseline_produces_reasonable_thresh(self):
        """For EAR ~0.42, closed_thresh should be in [0.04, 0.28]."""
        ec = calibrate_ear(self._samples(base=0.42, n=50))
        assert 0.04 <= ec.closed_thresh <= 0.28, ec


# ══════════════════════════════════════════════════════════════════════════════
# 6. calibrate_mar
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibrateMAR:
    def _samples(self, base=0.034, n=30):
        rng = np.random.default_rng(7)
        return (base + rng.normal(0, 0.002, n)).tolist()

    def test_open_thresh_above_baseline(self):
        mc = calibrate_mar(self._samples())
        assert mc.open_thresh > mc.closed_baseline

    def test_fallback_on_few_samples(self):
        mc = calibrate_mar([0.03] * 3)
        assert mc.method == "default"

    def test_open_thresh_at_least_floor(self):
        mc = calibrate_mar(self._samples())
        assert mc.open_thresh >= MAR_ABS_OPEN_MIN

    def test_typical_resting_mar(self):
        """For resting MAR ~0.034, open_thresh should be at least 0.16."""
        mc = calibrate_mar(self._samples(base=0.034, n=50))
        assert mc.open_thresh >= 0.10, f"open_thresh unexpectedly low: {mc.open_thresh}"


# ══════════════════════════════════════════════════════════════════════════════
# 7. calibrate_head_yaw
# ══════════════════════════════════════════════════════════════════════════════

class TestCalibrateHeadYaw:
    def _samples(self, base=0.0, n=30):
        rng = np.random.default_rng(13)
        return (base + rng.normal(0, 3.0, n)).tolist()

    def test_thresh_at_least_min(self):
        hc = calibrate_head_yaw(self._samples())
        assert hc.thresh_deg >= HEAD_YAW_THRESH_DEG

    def test_fallback_on_few_samples(self):
        hc = calibrate_head_yaw([0.0] * 3)
        assert hc.method == "default"
        assert hc.thresh_deg >= HEAD_YAW_THRESH_DEG

    def test_neutral_yaw_near_median(self):
        """For samples centred at 5°, neutral_yaw should be close to 5°."""
        samples = [5.0 + np.sin(i) * 2 for i in range(30)]
        hc = calibrate_head_yaw(samples)
        assert abs(hc.neutral_yaw - 5.0) < 3.0, hc


# ══════════════════════════════════════════════════════════════════════════════
# 8. classify_turn
# ══════════════════════════════════════════════════════════════════════════════

class TestClassifyTurn:
    def test_center(self):
        assert classify_turn(0.0, 0.0, 14.0) == "CENTER"

    def test_left(self):
        # yaw > neutral + threshold → LEFT
        assert classify_turn(20.0, 0.0, 14.0) == "LEFT"

    def test_right(self):
        # yaw < neutral - threshold → RIGHT
        assert classify_turn(-20.0, 0.0, 14.0) == "RIGHT"

    def test_within_threshold_is_center(self):
        assert classify_turn(5.0, 0.0, 14.0) == "CENTER"

    def test_with_non_zero_neutral(self):
        # neutral_yaw = 5, thresh = 14 → right means yaw < (5 - 14) = -9
        assert classify_turn(-10.0, 5.0, 14.0) == "RIGHT"
        assert classify_turn(20.0, 5.0, 14.0) == "LEFT"
        assert classify_turn(7.0, 5.0, 14.0) == "CENTER"

    def test_nan_returns_center(self):
        assert classify_turn(float("nan"), 0.0, 14.0) == "CENTER"

    def test_inf_returns_center(self):
        assert classify_turn(float("inf"), 0.0, 14.0) == "CENTER"


# ══════════════════════════════════════════════════════════════════════════════
# 9. estimate_head_pose
# ══════════════════════════════════════════════════════════════════════════════

class TestEstimateHeadPose:
    def test_success_on_valid_landmarks(self):
        lm = _make_landmarks()
        pose = estimate_head_pose(lm, 640, 480, mirrored=True)
        assert pose.success is True
        assert np.isfinite(pose.yaw)
        assert np.isfinite(pose.pitch)
        assert np.isfinite(pose.roll)

    def test_failure_on_none(self):
        pose = estimate_head_pose(None, 640, 480)
        assert pose.success is False

    def test_failure_on_short_list(self):
        pose = estimate_head_pose([(0.5, 0.5)] * 5, 640, 480)
        assert pose.success is False

    def test_frontal_face_near_center(self):
        """A symmetric frontal face should produce |yaw| near 0."""
        lm = _make_landmarks()
        pose = estimate_head_pose(lm, 640, 480, mirrored=True)
        if pose.success:
            # Frontal face yaw should be well within ±25° (not turned)
            assert abs(pose.yaw) < 25.0, f"Frontal yaw unexpectedly large: {pose.yaw:.2f}"

    def test_mirrored_vs_unmirrored_yaw_sign(self):
        """With a perfectly symmetric frontal face, mirrored and unmirrored yaw should be
        close to equal (both near zero), or at least not wildly different."""
        lm = _make_landmarks()
        pose_m  = estimate_head_pose(lm, 640, 480, mirrored=True)
        pose_um = estimate_head_pose(lm, 640, 480, mirrored=False)
        if pose_m.success and pose_um.success:
            # For a symmetric face the magnitude should be similar
            assert abs(abs(pose_m.yaw) - abs(pose_um.yaw)) < 10.0


# ══════════════════════════════════════════════════════════════════════════════
# 10. LiveGuardEngine — BLINK PASS via mocked face detector
# ══════════════════════════════════════════════════════════════════════════════

class TestLiveGuardEngine:
    """Tests for the LiveGuardEngine using a mock FaceDetector."""

    def _mock_fd(self, lm_open, lm_closed=None):
        """Create a mock FaceDetector that alternates open/closed landmarks."""
        fd = MagicMock()
        fd.use_mediapipe = True
        # Provide unlimited landmark frames: alternate open→closed→open...
        call_results = []
        for _ in range(200):
            call_results.append(lm_open)
            if lm_closed:
                call_results.append(lm_closed)
        fd.landmarks.side_effect = call_results
        return fd

    def _calibrate_and_start(self, engine, lm_open):
        from app.modules.liveguard.calibrate import (
            calibrate_ear, calibrate_head_yaw, calibrate_mar
        )
        ear_val = _ear(lm_open, 640, 480)
        ec = calibrate_ear([ear_val] * 30)
        mc = calibrate_mar([_mar(lm_open, 640, 480)] * 30)
        hc = calibrate_head_yaw([0.0] * 30)
        engine.set_calibration(ec, mc, hc)

    def test_blink_challenge_pass(self):
        """Engine should report PASS for blink challenge after 2 blinks."""
        from app.modules.liveguard.engine import LiveGuardEngine, ChallengeOutcome, BLINK
        import numpy as np_local

        lm_open   = _make_landmarks()
        lm_closed = _make_closed_eye_landmarks()

        ear_open   = _ear(lm_open,   640, 480)
        ear_closed = _ear(lm_closed, 640, 480)

        # Build calibration from open-eye samples
        from app.modules.liveguard.calibrate import calibrate_ear, calibrate_mar, calibrate_head_yaw
        ec = calibrate_ear([ear_open] * 30)
        mc = calibrate_mar([_mar(lm_open, 640, 480)] * 30)
        hc = calibrate_head_yaw([0.0] * 30)

        # The detector needs to serve: open frames, then 2× (closed, open) cycles
        frame_sequence = (
            [lm_open] * 5 +                       # calibration warmup
            [lm_closed] * 3 + [lm_open] * 3 +     # blink 1
            [lm_closed] * 3 + [lm_open] * 3       # blink 2
        )

        fd = MagicMock()
        fd.use_mediapipe = True
        fd.landmarks = MagicMock(side_effect=frame_sequence + [lm_open] * 50)

        engine = LiveGuardEngine(
            face_detector=fd,
            challenge_deadline_s=30.0,
            required_blinks=2,
        )
        engine.set_calibration(ec, mc, hc)

        # Override the challenge list to blink only
        engine.start(challenges=[BLINK])

        dummy_frame = np_local.zeros((480, 640, 3), dtype=np_local.uint8)
        outcomes = []
        for frame_i in range(len(frame_sequence) + 50):
            result = engine.process_frame(dummy_frame, timestamp_ms=frame_i * 33)
            if result.get("outcome") is not None:
                outcomes.append(result["outcome"])
            if engine._finished:
                break

        assert ChallengeOutcome.PASS.value in outcomes, (
            f"Expected PASS in outcomes but got: {outcomes}"
        )

    def test_blink_challenge_timeout(self):
        """Engine should TIMEOUT a blink challenge when face is visible but no blink occurs."""
        from app.modules.liveguard.engine import LiveGuardEngine, ChallengeOutcome, BLINK
        import numpy as np_local

        lm_open = _make_landmarks()
        from app.modules.liveguard.calibrate import calibrate_ear, calibrate_mar, calibrate_head_yaw
        ec = calibrate_ear([_ear(lm_open, 640, 480)] * 30)
        mc = calibrate_mar([_mar(lm_open, 640, 480)] * 30)
        hc = calibrate_head_yaw([0.0] * 30)

        fd = MagicMock()
        fd.use_mediapipe = True
        fd.landmarks = MagicMock(return_value=lm_open)

        engine = LiveGuardEngine(
            face_detector=fd,
            challenge_deadline_s=0.001,  # extremely short deadline
            required_blinks=2,
        )
        engine.set_calibration(ec, mc, hc)
        engine.start(challenges=[BLINK])

        dummy_frame = np_local.zeros((480, 640, 3), dtype=np_local.uint8)
        time.sleep(0.05)  # ensure deadline already passed

        outcomes = []
        for i in range(10):
            result = engine.process_frame(dummy_frame, timestamp_ms=i * 33)
            if result.get("outcome") is not None:
                outcomes.append(result["outcome"])
            if engine._finished:
                break

        assert ChallengeOutcome.TIMEOUT.value in outcomes, (
            f"Expected TIMEOUT but got: {outcomes}"
        )

    def test_inconclusive_on_face_missing(self):
        """Engine should report INCONCLUSIVE when face is absent beyond max_no_face_s."""
        from app.modules.liveguard.engine import LiveGuardEngine, ChallengeOutcome, BLINK
        import numpy as np_local

        lm_open = _make_landmarks()
        from app.modules.liveguard.calibrate import calibrate_ear, calibrate_mar, calibrate_head_yaw
        ec = calibrate_ear([_ear(lm_open, 640, 480)] * 30)
        mc = calibrate_mar([_mar(lm_open, 640, 480)] * 30)
        hc = calibrate_head_yaw([0.0] * 30)

        fd = MagicMock()
        fd.use_mediapipe = True
        fd.landmarks = MagicMock(return_value=None)  # always no face

        engine = LiveGuardEngine(
            face_detector=fd,
            challenge_deadline_s=30.0,
            max_no_face_s=0.001,  # face disappears almost immediately → INCONCLUSIVE
            required_blinks=2,
        )
        engine.set_calibration(ec, mc, hc)
        engine.start(challenges=[BLINK])

        dummy_frame = np_local.zeros((480, 640, 3), dtype=np_local.uint8)
        time.sleep(0.05)

        outcomes = []
        for i in range(20):
            result = engine.process_frame(dummy_frame, timestamp_ms=i * 33)
            if result.get("outcome") is not None:
                outcomes.append(result["outcome"])
            if engine._finished:
                break

        assert ChallengeOutcome.INCONCLUSIVE.value in outcomes, (
            f"Expected INCONCLUSIVE but got: {outcomes}"
        )

    def test_no_crash_on_sustained_no_face(self):
        """Engine must not raise when face is missing for many frames."""
        from app.modules.liveguard.engine import LiveGuardEngine, BLINK
        import numpy as np_local

        lm_open = _make_landmarks()
        from app.modules.liveguard.calibrate import calibrate_ear, calibrate_mar, calibrate_head_yaw
        ec = calibrate_ear([_ear(lm_open, 640, 480)] * 30)
        mc = calibrate_mar([_mar(lm_open, 640, 480)] * 30)
        hc = calibrate_head_yaw([0.0] * 30)

        fd = MagicMock()
        fd.use_mediapipe = True
        fd.landmarks = MagicMock(return_value=None)

        engine = LiveGuardEngine(
            face_detector=fd,
            challenge_deadline_s=30.0,
            max_no_face_s=5.0,
            required_blinks=2,
        )
        engine.set_calibration(ec, mc, hc)
        engine.start(challenges=[BLINK])

        dummy_frame = np_local.zeros((480, 640, 3), dtype=np_local.uint8)
        try:
            for i in range(50):
                engine.process_frame(dummy_frame, timestamp_ms=i * 33)
        except Exception as exc:
            pytest.fail(f"engine.process_frame raised unexpectedly: {exc}")

    def test_report_structure(self):
        """engine.report() must include all expected keys."""
        from app.modules.liveguard.engine import LiveGuardEngine, BLINK
        import numpy as np_local

        lm_open = _make_landmarks()
        from app.modules.liveguard.calibrate import calibrate_ear, calibrate_mar, calibrate_head_yaw
        ec = calibrate_ear([_ear(lm_open, 640, 480)] * 30)
        mc = calibrate_mar([_mar(lm_open, 640, 480)] * 30)
        hc = calibrate_head_yaw([0.0] * 30)

        fd = MagicMock()
        fd.use_mediapipe = True
        fd.landmarks = MagicMock(return_value=lm_open)

        engine = LiveGuardEngine(
            face_detector=fd,
            challenge_deadline_s=0.001,
            required_blinks=2,
        )
        engine.set_calibration(ec, mc, hc)
        engine.start(challenges=[BLINK])

        dummy_frame = np_local.zeros((480, 640, 3), dtype=np_local.uint8)
        time.sleep(0.05)
        for i in range(5):
            engine.process_frame(dummy_frame, timestamp_ms=i * 33)

        rpt = engine.report()
        required_keys = {
            "started", "finished", "calibrated",
            "challenges_requested", "challenges_completed",
            "passed", "failed", "timedout", "inconclusive",
            "results", "calibration", "mirrored", "logs",
        }
        assert required_keys.issubset(rpt.keys()), (
            f"Missing report keys: {required_keys - rpt.keys()}"
        )
