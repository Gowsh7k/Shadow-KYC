"""LiveGuard active liveness challenge components.

This package reuses :class:`app.modules.face_detector.FaceDetector` (a single
detector class — no duplicate implementation). It provides calibrated,
privacy-aware active liveness challenges (blink, mouth open, head turn) with
clear PASS / FAIL / TIMEOUT / INCONCLUSIVE outcomes.

Security note: successful challenges are liveness *indicators only*. They are
never interpreted as proof against deepfakes, replay attacks or virtual cameras.
"""
from .engine import LiveGuardEngine, ChallengeOutcome
from .diagnostic import build_diagnostic_report

__all__ = ["LiveGuardEngine", "ChallengeOutcome", "build_diagnostic_report"]
