"""
Calibrated 3D head pose via solvePnP.

MediaPipe FaceLandmarker provides 478 normalized landmarks. We solve PnP with a
small set of well-separated facial points and a pinhole camera model, then
convert the rotation vector to yaw/pitch/roll degrees.

cv2.RQDecomp3x3 returns three RQ-decomposition angles in degrees.
For the rotation matrix produced by cv2.Rodrigues applied to a solvePnP result:
    angles[0]  ≈  pitch  (X-axis rotation)
    angles[1]  ≈  yaw    (Y-axis rotation)
    angles[2]  ≈  roll   (Z-axis rotation)

Sign convention (with mirrored=True, the default for selfie/webcam):
    * Positive yaw  = subject's head turned to their LEFT  (image-right on mirrored cam)
    * Negative yaw  = subject's head turned to their RIGHT (image-left on mirrored cam)
    * mirrored=True flips the x-pixel coordinate before solvePnP to compensate for
      the horizontal mirror inherent in most webcam previews.
    * If your camera feed is NOT mirrored, pass mirrored=False (or flip yaw_sign=-1).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

# MediaPipe FaceLandmarker landmark indices used for pose estimation.
# Stable across the FaceLandmarker model (478 face landmarks).
_LM_NOSE_TIP = 1
_LM_CHIN = 152
_LM_LEFT_EYE_OUTER = 33   # subject's left eye (image-left when not mirrored)
_LM_RIGHT_EYE_OUTER = 263
_LM_LEFT_MOUTH = 61
_LM_RIGHT_MOUTH = 291

# 3D model points (mm) in a right-handed face-centric coordinate system:
#   X -> subject's right, Y -> up, Z -> forward (toward camera).
_MODEL_POINTS = np.array([
    (0.0,   0.0,   0.0),   # nose tip
    (0.0,  -63.0,  -4.0),  # chin
    (-42.0,  32.0, -26.0), # left eye outer
    (42.0,   32.0, -26.0), # right eye outer
    (-38.0,  56.0, -10.0), # left mouth corner
    (38.0,   56.0, -10.0), # right mouth corner
], dtype="double")


@dataclass
class HeadPose:
    yaw: float  # degrees, + = subject's left
    pitch: float  # degrees
    roll: float  # degrees
    success: bool
    neutral_delta: Optional[float] = None  # yaw - neutral_yaw


def _camera_matrix(w: int, h: int) -> np.ndarray:
    # Approximate pinhole model: focal length ~ image width, principal point
    # at centre. Good enough for direction (left/right) classification.
    fl = float(w)
    return np.array([
        [fl, 0, w / 2.0],
        [0, fl, h / 2.0],
        [0, 0, 1.0],
    ], dtype="double")


def estimate_head_pose(
    landmarks: list,
    w: int,
    h: int,
    mirrored: bool = True,
    yaw_sign: float = 1.0,
) -> HeadPose:
    """Estimate head pose from normalized landmarks.

    Parameters
    ----------
    landmarks : list of (x, y) normalized (0..1) tuples.
    w, h : image width / height in pixels.
    mirrored : if True, horizontally flip image x before solving (selfie cam).
    yaw_sign : multiplier applied to the resulting yaw (+1 or -1) to match the
        desired sign convention.
    """
    if landmarks is None or len(landmarks) <= max(_LM_RIGHT_MOUTH, 152):
        return HeadPose(0.0, 0.0, 0.0, False)

    try:

        def pt(idx):
            x, y = landmarks[idx]
            x_px = x * w
            y_px = y * h
            if mirrored:
                x_px = w - x_px
            return (x_px, y_px)

        image_points = np.array([
            pt(_LM_NOSE_TIP),
            pt(_LM_CHIN),
            pt(_LM_LEFT_EYE_OUTER),
            pt(_LM_RIGHT_EYE_OUTER),
            pt(_LM_LEFT_MOUTH),
            pt(_LM_RIGHT_MOUTH),
        ], dtype="double")

        cam = _camera_matrix(w, h)
        dist_coeffs = np.zeros((4, 1))
        ok, rvec, _ = cv2.solvePnP(
            _MODEL_POINTS, image_points, cam, dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            return HeadPose(0.0, 0.0, 0.0, False)

        rmat, _ = cv2.Rodrigues(rvec)
        # cv2.RQDecomp3x3 returns 6 values:
        #   (angles_3d, mtxR, mtxQ, Qx, Qy, Qz)
        # where angles_3d is a (3,) array in degrees:
        #   angles_3d[0] ~ pitch (X-axis rotation)
        #   angles_3d[1] ~ yaw   (Y-axis rotation)
        #   angles_3d[2] ~ roll  (Z-axis rotation)
        # Not pure Euler angles but a good approximation for small rotations.
        angles, _mtxR, _mtxQ, _Qx, _Qy, _Qz = cv2.RQDecomp3x3(rmat)
        pitch = float(angles[0])
        yaw   = float(angles[1])
        roll  = float(angles[2])
        yaw   = yaw * yaw_sign

        return HeadPose(yaw=yaw, pitch=pitch, roll=roll, success=True)
    except Exception:
        return HeadPose(0.0, 0.0, 0.0, False)


def classify_turn(yaw: float, neutral_yaw: float, threshold_deg: float) -> str:
    """Classify head turn relative to a neutral reference.

    Returns 'LEFT', 'RIGHT' or 'CENTER'.
    """
    if not np.isfinite(yaw) or not np.isfinite(neutral_yaw):
        return "CENTER"
    delta = yaw - neutral_yaw
    if delta > threshold_deg:
        return "LEFT"
    if delta < -threshold_deg:
        return "RIGHT"
    return "CENTER"
