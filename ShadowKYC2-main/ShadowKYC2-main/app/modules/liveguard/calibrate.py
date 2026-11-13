"""
Calibration helpers for LiveGuard active-liveness signals.

Thresholds are derived from the observed signal distribution during a short
warm-up rather than hard-coded to arbitrary constants. This avoids the classic
failure mode where a fixed EAR/MAR limit never matches a particular camera,
face size or lighting condition.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np


# --- EAR (eye aspect ratio) calibration -------------------------------------
# Eyes are "open" most of the time, so the open baseline is taken from the
# upper part of the EAR distribution. A blink is a drop below closed_thresh
# followed by a recovery above reopen_thresh (hysteresis).
EAR_CLOSED_RATIO = 0.50   # closed threshold as fraction of open baseline
EAR_REOPEN_RATIO = 0.80   # re-open threshold as fraction of open baseline
EAR_ABS_CLOSED_MIN = 0.04  # never go below this (avoid divide-by-zero noise)
EAR_ABS_CLOSED_MAX = 0.28  # never go above this (sanity cap)
EAR_MIN_SAMPLES = 8       # need at least this many frames to calibrate


# --- MAR (mouth aspect ratio) calibration -----------------------------------
# Mouth is "closed" most of the time, so the closed baseline is the median.
# An intentional open-mouth must exceed closed_baseline by a relative margin.
MAR_OPEN_MULTIPLIER = 2.6   # open threshold relative to closed baseline
MAR_OPEN_DELTA = 0.09       # absolute delta added on top
MAR_ABS_OPEN_MIN = 0.16     # sanity floor for the open threshold
MAR_MIN_SAMPLES = 8


# --- Head yaw calibration ----------------------------------------------------
HEAD_YAW_THRESH_DEG = 14.0  # degrees from neutral to count as a turn
HEAD_YAW_MIN_SPREAD_DEG = 6.0  # minimum spread to trust the neutral estimate
HEAD_MIN_SAMPLES = 8


@dataclass
class EARCalibration:
    open_baseline: float
    closed_thresh: float
    reopen_thresh: float
    n_samples: int
    method: str = "percentile"


@dataclass
class MARCalibration:
    closed_baseline: float
    open_thresh: float
    n_samples: int
    method: str = "median"


@dataclass
class HeadYawCalibration:
    neutral_yaw: float
    thresh_deg: float
    n_samples: int
    method: str = "median"


def _finite(values: List[float]) -> List[float]:
    return [float(v) for v in values if v is not None and np.isfinite(v)]


def calibrate_ear(ear_samples: List[float]) -> EARCalibration:
    """Derive adaptive EAR closed / re-open thresholds."""
    vals = _finite(ear_samples)
    if len(vals) < EAR_MIN_SAMPLES:
        # Not enough data: fall back to conservative defaults so the detector
        # reports INCONCLUSIVE rather than fabricating a blink threshold.
        return EARCalibration(
            open_baseline=0.30,
            closed_thresh=0.15,
            reopen_thresh=0.24,
            n_samples=len(vals),
            method="default",
        )
    arr = np.array(vals, dtype=float)
    # Open baseline: eyes are open most frames -> upper quartile.
    open_baseline = float(np.percentile(arr, 75))
    closed_thresh = open_baseline * EAR_CLOSED_RATIO
    reopen_thresh = open_baseline * EAR_REOPEN_RATIO
    # Sanity clamps so thresholds stay in a plausible EAR range.
    closed_thresh = float(np.clip(closed_thresh, EAR_ABS_CLOSED_MIN, EAR_ABS_CLOSED_MAX))
    reopen_thresh = float(np.clip(reopen_thresh, closed_thresh + 0.02, 0.5))
    return EARCalibration(
        open_baseline=round(open_baseline, 4),
        closed_thresh=round(closed_thresh, 4),
        reopen_thresh=round(reopen_thresh, 4),
        n_samples=len(vals),
    )


def calibrate_mar(mar_samples: List[float]) -> MARCalibration:
    """Derive an adaptive MAR open threshold."""
    vals = _finite(mar_samples)
    if len(vals) < MAR_MIN_SAMPLES:
        return MARCalibration(
            closed_baseline=0.04,
            open_thresh=0.20,
            n_samples=len(vals),
            method="default",
        )
    arr = np.array(vals, dtype=float)
    closed_baseline = float(np.median(arr))  # mouth closed most of the time
    open_thresh = closed_baseline * MAR_OPEN_MULTIPLIER + MAR_OPEN_DELTA
    open_thresh = float(np.clip(open_thresh, MAR_ABS_OPEN_MIN, 1.0))
    return MARCalibration(
        closed_baseline=round(closed_baseline, 4),
        open_thresh=round(open_thresh, 4),
        n_samples=len(vals),
    )


def calibrate_head_yaw(yaw_samples: List[float]) -> HeadYawCalibration:
    """Derive neutral yaw and a per-subject turn threshold."""
    vals = _finite(yaw_samples)
    if len(vals) < HEAD_MIN_SAMPLES:
        return HeadYawCalibration(
            neutral_yaw=0.0,
            thresh_deg=HEAD_YAW_THRESH_DEG,
            n_samples=len(vals),
            method="default",
        )
    arr = np.array(vals, dtype=float)
    neutral_yaw = float(np.median(arr))
    # Robust spread: interquartile range / 1.35 approximates sigma.
    q1, q3 = np.percentile(arr, [25, 75])
    spread = float(q3 - q1)
    thresh_deg = max(HEAD_YAW_THRESH_DEG, HEAD_YAW_MIN_SPREAD_DEG, spread * 1.5 + 4.0)
    return HeadYawCalibration(
        neutral_yaw=round(neutral_yaw, 2),
        thresh_deg=round(thresh_deg, 2),
        n_samples=len(vals),
    )
