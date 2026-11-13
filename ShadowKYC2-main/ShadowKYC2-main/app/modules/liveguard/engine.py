from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

import cv2
import numpy as np

from app.core import config
from app.modules.face_detector import FaceDetector
from .blink import BlinkCounter
from .calibrate import (
    EARCalibration,
    HeadYawCalibration,
    MARCalibration,
    calibrate_ear,
    calibrate_head_yaw,
    calibrate_mar,
)
from .geometry import HeadPose, classify_turn, estimate_head_pose


class ChallengeOutcome(str, Enum):
    PASS = "PASS"
    FAIL = "FAIL"
    TIMEOUT = "TIMEOUT"
    INCONCLUSIVE = "INCONCLUSIVE"


BLINK = "blink"
OPEN_MOUTH = "open_mouth"
TURN_HEAD_LEFT = "turn_head_left"
TURN_HEAD_RIGHT = "turn_head_right"

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
LM_TOP_LIP = 13
LM_BOTTOM_LIP = 14
LM_MOUTH_LEFT = 78
LM_MOUTH_RIGHT = 308

DEFAULT_CHALLENGES = [BLINK, OPEN_MOUTH, TURN_HEAD_LEFT, TURN_HEAD_RIGHT]


def _ear(landmarks: List[tuple], w: int, h: int) -> float:
    def eye(ids):
        pts = [(landmarks[i][0] * w, landmarks[i][1] * h) for i in ids]
        A = float(np.linalg.norm(np.array(pts[1]) - np.array(pts[5])))
        B = float(np.linalg.norm(np.array(pts[2]) - np.array(pts[4])))
        C = float(np.linalg.norm(np.array(pts[0]) - np.array(pts[3])))
        return (A + B) / (2.0 * C + 1e-6)

    return float((eye(LEFT_EYE) + eye(RIGHT_EYE)) / 2.0)


def _mar(landmarks: List[tuple], w: int, h: int) -> float:
    top = np.array([landmarks[LM_TOP_LIP][0] * w, landmarks[LM_TOP_LIP][1] * h])
    bottom = np.array([landmarks[LM_BOTTOM_LIP][0] * w, landmarks[LM_BOTTOM_LIP][1] * h])
    left = np.array([landmarks[LM_MOUTH_LEFT][0] * w, landmarks[LM_MOUTH_LEFT][1] * h])
    right = np.array([landmarks[LM_MOUTH_RIGHT][0] * w, landmarks[LM_MOUTH_RIGHT][1] * h])
    vertical = float(np.linalg.norm(top - bottom))
    horizontal = float(np.linalg.norm(left - right))
    return float(vertical / max(horizontal, 1e-6))


@dataclass
class LiveGuardEngine:
    face_detector: FaceDetector
    mirrored: bool = True
    challenge_deadline_s: float = 8.0
    max_no_face_s: float = 5.0
    required_blinks: int = 2
    mouth_sustain_frames: int = 3
    head_sustain_frames: int = 3
    challenges: List[str] = field(default_factory=lambda: list(DEFAULT_CHALLENGES))

    ear_cal: Optional[EARCalibration] = field(default=None, init=False)
    mar_cal: Optional[MARCalibration] = field(default=None, init=False)
    head_cal: Optional[HeadYawCalibration] = field(default=None, init=False)
    neutral_yaw: float = field(default=0.0, init=False)

    _challenge_list: List[str] = field(default_factory=list, init=False)
    _challenge_index: int = field(default=0, init=False)
    _current: Optional[dict] = field(default=None, init=False)
    _results: List[dict] = field(default_factory=list, init=False)
    _blink: Optional[BlinkCounter] = field(default=None, init=False)
    _mouth_sustain: int = field(default=0, init=False)
    _head_sustain: int = field(default=0, init=False)
    _frame_index: int = field(default=0, init=False)
    _started: bool = field(default=False, init=False)
    _finished: bool = field(default=False, init=False)
    _no_face_start: Optional[float] = field(default=None, init=False)
    _no_face_total: float = field(default=0.0, init=False)
    _consecutive_no_face: int = field(default=0, init=False)
    _last_frame_ts: Optional[float] = field(default=None, init=False)
    _logs: List[str] = field(default_factory=list, init=False)

    def log(self, message: str) -> None:
        self._logs.append(message)
        print("[LiveGuard] " + message)

    def set_calibration(self, ear_cal, mar_cal, head_cal) -> None:
        self.ear_cal = ear_cal
        self.mar_cal = mar_cal
        self.head_cal = head_cal
        self.neutral_yaw = head_cal.neutral_yaw
        self._blink = BlinkCounter(required=self.required_blinks)
        self._blink.detector.ear_closed_thresh = ear_cal.closed_thresh
        self._blink.detector.ear_reopen_thresh = ear_cal.reopen_thresh
        self.log("calibration set: ear_closed=%.3f ear_reopen=%.3f mar_open=%.3f "
                 "yaw_neutral=%.2f yaw_thresh=%.2f" % (
                     ear_cal.closed_thresh, ear_cal.reopen_thresh,
                     mar_cal.open_thresh, head_cal.neutral_yaw,
                     head_cal.thresh_deg))

    def is_calibrated(self) -> bool:
        return (self.ear_cal is not None and self.mar_cal is not None
                and self.head_cal is not None)

    def start(self, challenges: Optional[List[str]] = None) -> None:
        self._challenge_list = list(challenges or self.challenges)
        random.shuffle(self._challenge_list)
        self._challenge_index = 0
        self._current = None
        self._results = []
        self._blink = BlinkCounter(required=self.required_blinks)
        if self.is_calibrated():
            self._blink.detector.ear_closed_thresh = self.ear_cal.closed_thresh
            self._blink.detector.ear_reopen_thresh = self.ear_cal.reopen_thresh
        self._mouth_sustain = 0
        self._head_sustain = 0
        self._frame_index = 0
        self._started = True
        self._finished = False
        self._no_face_start = None
        self._no_face_total = 0.0
        self._last_frame_ts = None
        self._logs = []
        self.log("started %d challenges (randomised order)" % len(self._challenge_list))
        self._advance()

    def _advance(self) -> None:
        if self._challenge_index >= len(self._challenge_list):
            self._current = None
            self._finished = True
            self.log("all challenges complete")
            return
        ctype = self._challenge_list[self._challenge_index]
        self._current = {
            "type": ctype,
            "index": self._challenge_index,
            "total": len(self._challenge_list),
            "start_ts": time.monotonic(),
            "deadline": self.challenge_deadline_s,
            "outcome": None,
        }
        self._blink.reset(self.required_blinks)
        self._mouth_sustain = 0
        self._head_sustain = 0
        self._no_face_start = None
        self.log("challenge %d/%d: %s (deadline %.0fs)" % (
            self._challenge_index + 1, len(self._challenge_list), ctype.upper(),
            self.challenge_deadline_s))

    def _complete(self, outcome: ChallengeOutcome, detail: str = "") -> None:
        cur = self._current
        if cur is None:
            return
        cur["outcome"] = outcome.value
        cur["ended_ts"] = time.monotonic()
        cur["duration_s"] = round(cur["ended_ts"] - cur["start_ts"], 3)
        cur["detail"] = detail
        self._results.append(dict(cur))
        self.log("challenge %s -> %s %s" % (cur["type"], outcome.value, detail))
        self._challenge_index += 1
        self._current = None
        self._advance()
    def _deadline_remaining(self, cur: dict, now: float) -> float:
        face_lost = self._no_face_total
        if self._no_face_start is not None:
            face_lost += now - self._no_face_start
        elapsed = (now - cur["start_ts"]) - face_lost
        return cur["deadline"] - elapsed

    def process_frame(
        self,
        frame_bgr: np.ndarray,
        timestamp_ms: Optional[int] = None,
    ) -> dict:
        self._frame_index += 1
        now = time.monotonic()
        if timestamp_ms is None:
            timestamp_ms = int(now * 1000)

        result: Dict = {
            "frame_index": self._frame_index,
            "face_detected": False,
            "ear": None,
            "mar": None,
            "head_pose": {"yaw": 0.0, "pitch": 0.0, "roll": 0.0, "success": False},
            "challenge": None,
            "challenge_status": "idle",
            "outcome": None,
            "blink_count": 0,
            "deadline_remaining_s": None,
            "calibrated": self.is_calibrated(),
            "logs": [],
        }

        if not self._started or self._finished:
            result["challenge_status"] = "finished" if self._finished else "not_started"
            return result

        cur = self._current
        landmarks = self.face_detector.landmarks(frame_bgr, timestamp_ms=timestamp_ms)
        h, w = frame_bgr.shape[:2]

        if landmarks is None:
            result["face_detected"] = False
            if self._no_face_start is None:
                self._no_face_start = now
            else:
                self._no_face_total += now - (self._last_frame_ts or now)
            self._last_frame_ts = now
            self._consecutive_no_face += 1
            if self._consecutive_no_face >= 3 or self._no_face_total >= self.max_no_face_s:
                self._complete(ChallengeOutcome.INCONCLUSIVE,
                               "face missing too long")
                result["challenge_status"] = "inconclusive"
                result["outcome"] = ChallengeOutcome.INCONCLUSIVE.value
                return result
            result["challenge_status"] = "active"
            result["deadline_remaining_s"] = self._deadline_remaining(cur, now)
            return result

        result["face_detected"] = True
        self._no_face_total = 0.0
        self._no_face_start = None
        self._consecutive_no_face = 0
        self._last_frame_ts = now

        ear = _ear(landmarks, w, h)
        mar = _mar(landmarks, w, h)
        pose: HeadPose = estimate_head_pose(landmarks, w, h, mirrored=self.mirrored)
        result["ear"] = round(ear, 4)
        result["mar"] = round(mar, 4)
        result["head_pose"] = {
            "yaw": round(pose.yaw, 2),
            "pitch": round(pose.pitch, 2),
            "roll": round(pose.roll, 2),
            "success": pose.success,
        }

        ctype = cur["type"]
        passed = False

        if ctype == BLINK:
            if self._blink.update(ear):
                passed = True
            result["blink_count"] = self._blink.detector.blink_count

        elif ctype == OPEN_MOUTH:
            if self.mar_cal is not None and mar >= self.mar_cal.open_thresh:
                self._mouth_sustain += 1
            else:
                self._mouth_sustain = 0
            if self._mouth_sustain >= self.mouth_sustain_frames:
                passed = True

        elif ctype in (TURN_HEAD_LEFT, TURN_HEAD_RIGHT):
            if pose.success:
                direction = classify_turn(
                    pose.yaw, self.neutral_yaw, self.head_cal.thresh_deg
                )
                wanted = "LEFT" if ctype == TURN_HEAD_LEFT else "RIGHT"
                if direction == wanted:
                    self._head_sustain += 1
                else:
                    self._head_sustain = 0
                if self._head_sustain >= self.head_sustain_frames:
                    passed = True

        if passed:
            self._complete(ChallengeOutcome.PASS, "signal sustained")
            result["challenge_status"] = "passed"
            result["outcome"] = ChallengeOutcome.PASS.value
            return result

        remaining = self._deadline_remaining(cur, now)
        result["deadline_remaining_s"] = round(remaining, 2)
        if remaining <= 0:
            self._complete(ChallengeOutcome.TIMEOUT, "deadline exceeded")
            result["challenge_status"] = "timedout"
            result["outcome"] = ChallengeOutcome.TIMEOUT.value
            return result

        result["challenge_status"] = "active"
        result["challenge"] = {
            "type": ctype,
            "index": cur["index"],
            "total": cur["total"],
        }
        return result

    def report(self) -> dict:
        passed = sum(1 for r in self._results if r["outcome"] == ChallengeOutcome.PASS.value)
        timedout = sum(1 for r in self._results if r["outcome"] == ChallengeOutcome.TIMEOUT.value)
        inconclusive = sum(1 for r in self._results if r["outcome"] == ChallengeOutcome.INCONCLUSIVE.value)
        failed = sum(1 for r in self._results if r["outcome"] == ChallengeOutcome.FAIL.value)
        return {
            "started": self._started,
            "finished": self._finished,
            "calibrated": self.is_calibrated(),
            "challenges_requested": len(self._challenge_list),
            "challenges_completed": len(self._results),
            "passed": passed,
            "failed": failed,
            "timedout": timedout,
            "inconclusive": inconclusive,
            "results": self._results,
            "current_challenge": self._current,
            "calibration": {
                "ear": None if self.ear_cal is None else {
                    "open_baseline": self.ear_cal.open_baseline,
                    "closed_thresh": self.ear_cal.closed_thresh,
                    "reopen_thresh": self.ear_cal.reopen_thresh,
                    "n_samples": self.ear_cal.n_samples,
                },
                "mar": None if self.mar_cal is None else {
                    "closed_baseline": self.mar_cal.closed_baseline,
                    "open_thresh": self.mar_cal.open_thresh,
                    "n_samples": self.mar_cal.n_samples,
                },
                "head_yaw": None if self.head_cal is None else {
                    "neutral_yaw": self.head_cal.neutral_yaw,
                    "thresh_deg": self.head_cal.thresh_deg,
                    "n_samples": self.head_cal.n_samples,
                },
            },
            "mirrored": self.mirrored,
            "logs": list(self._logs),
        }
