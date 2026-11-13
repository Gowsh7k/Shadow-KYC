from __future__ import annotations

import numpy as np

from .calibrate import calibrate_ear, calibrate_head_yaw, calibrate_mar
from .engine import _ear, _mar
from .geometry import estimate_head_pose, classify_turn


def _synthetic_landmarks(w=640, h=480):
    cx, cy = 0.5, 0.55
    pts = [(cx, cy)] * 478
    pts[1] = (cx, cy - 0.02)
    pts[152] = (cx, cy + 0.18)
    pts[33] = (cx - 0.09, cy - 0.06)
    pts[263] = (cx + 0.09, cy - 0.06)
    pts[61] = (cx - 0.03, cy + 0.10)
    pts[291] = (cx + 0.03, cy + 0.10)
    for i in [160, 158, 153, 144]:
        pts[i] = (pts[33][0] + (i - 160) * 0.002, pts[33][1] + 0.02)
    for i in [385, 387, 373, 380]:
        pts[i] = (pts[263][0] - (i - 385) * 0.002, pts[263][1] + 0.02)
    return pts


def build_diagnostic_report(face_detector) -> dict:
    det_diag = face_detector.diagnostics()

    ear_samples, mar_samples, yaw_samples = [], [], []
    for _ in range(20):
        pts = _synthetic_landmarks()
        ear_samples.append(_ear(pts, 640, 480))
        mar_samples.append(_mar(pts, 640, 480))
        pose = estimate_head_pose(pts, 640, 480, mirrored=True)
        if pose.success:
            yaw_samples.append(pose.yaw)

    ear_cal = calibrate_ear(ear_samples)
    mar_cal = calibrate_mar(mar_samples)
    head_cal = calibrate_head_yaw(yaw_samples) if yaw_samples else None

    pose = estimate_head_pose(_synthetic_landmarks(), 640, 480, mirrored=True)
    h_cal = head_cal or type("HC", (), {"neutral_yaw": 0.0, "thresh_deg": 14.0})()
    direction = classify_turn(pose.yaw, h_cal.neutral_yaw, h_cal.thresh_deg)

    return {
        "detector": det_diag,
        "calibration": {
            "ear": {
                "open_baseline": ear_cal.open_baseline,
                "closed_thresh": ear_cal.closed_thresh,
                "reopen_thresh": ear_cal.reopen_thresh,
                "n_samples": ear_cal.n_samples,
                "method": ear_cal.method,
            },
            "mar": {
                "closed_baseline": mar_cal.closed_baseline,
                "open_thresh": mar_cal.open_thresh,
                "n_samples": mar_cal.n_samples,
                "method": mar_cal.method,
            },
            "head_yaw": None if head_cal is None else {
                "neutral_yaw": head_cal.neutral_yaw,
                "thresh_deg": head_cal.thresh_deg,
                "n_samples": head_cal.n_samples,
                "method": head_cal.method,
            },
        },
        "logic_check": {
            "ear_on_synthetic": round(_ear(_synthetic_landmarks(), 640, 480), 4),
            "mar_on_synthetic": round(_mar(_synthetic_landmarks(), 640, 480), 4),
            "head_pose_success": pose.success,
            "neutral_classifies_as_center": direction == "CENTER",
        },
        "note": "Active challenges require a client-side camera; run "
                "liveguard_liveness_test.py --camera. Successful challenges are "
                "liveness indicators only, not proof against deepfakes, replay "
                "attacks or virtual cameras.",
    }
