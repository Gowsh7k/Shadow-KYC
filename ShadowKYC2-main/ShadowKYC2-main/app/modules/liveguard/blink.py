from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class BlinkDetector:
    """Temporal blink detector with hysteresis.

    A blink is counted on the closed -> open transition only, so a sustained
    closure (eyes held shut) is counted once, not repeatedly.
    """

    ear_closed_thresh: float
    ear_reopen_thresh: float
    eyes_closed: bool = field(default=False, init=False)
    blink_count: int = field(default=0, init=False)

    def reset(self) -> None:
        self.eyes_closed = False
        self.blink_count = 0

    def update(self, ear: Optional[float]) -> bool:
        """Feed one EAR sample.

        Returns True exactly when a blink completes (eyes re-open after being
        closed). Returns False otherwise, including while eyes stay closed.
        """
        if ear is None:
            return False
        if ear < self.ear_closed_thresh and not self.eyes_closed:
            self.eyes_closed = True
            return False
        if ear >= self.ear_reopen_thresh and self.eyes_closed:
            self.eyes_closed = False
            self.blink_count += 1
            return True
        return False


@dataclass
class BlinkCounter:
    """Per-challenge blink counter wrapper."""

    required: int
    detector: BlinkDetector = field(init=False)
    passed: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        # Defaults are conservative estimates for a typical open-eye EAR of ~0.42.
        # The LiveGuardEngine always overrides these via calibrate_ear() before
        # the challenge starts; these are only used when called without calibration.
        self.detector = BlinkDetector(
            ear_closed_thresh=0.21,  # ~50% of 0.42 open baseline
            ear_reopen_thresh=0.28,  # ~67% of 0.42 open baseline
        )

    def reset(self, required: int) -> None:
        self.required = required
        self.detector.reset()
        self.passed = False

    def update(self, ear: Optional[float]) -> bool:
        self.detector.update(ear)
        if self.detector.blink_count >= self.required:
            self.passed = True
        return self.passed
