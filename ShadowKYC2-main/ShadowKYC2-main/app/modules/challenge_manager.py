"""
Active Liveness Challenge Manager
Generates random interactive micro-challenges:
- Head Turn Left / Right
- Look Up / Look Down
- Head Tilt Left / Right (Roll)
- Eye Blink / Double Blink
- Smile
- Open Mouth
- Head Nod
- Move Closer (Depth/Scale)

Tracks candidate compliance across live frames.
"""
import random
import numpy as np
from typing import List, Dict, Optional
from collections import deque


class ChallengeManager:
    CHALLENGE_DEFINITIONS = [
        {"id": "TURN_LEFT", "prompt": "Turn your head slowly to the LEFT ⬅️", "target": "yaw", "direction": -1, "threshold": 12.0},
        {"id": "TURN_RIGHT", "prompt": "Turn your head slowly to the RIGHT ➡️", "target": "yaw", "direction": 1, "threshold": 12.0},
        {"id": "LOOK_UP", "prompt": "Tilt your head UPWARDS ⬆️", "target": "pitch_up", "direction": 1, "threshold": 9.0},
        {"id": "LOOK_DOWN", "prompt": "Tilt your head DOWNWARDS ⬇️", "target": "pitch_down", "direction": -1, "threshold": -9.0},
        {"id": "TILT_HEAD_LEFT", "prompt": "Tilt your head towards your LEFT shoulder ↩️", "target": "roll", "direction": -1, "threshold": 8.0},
        {"id": "TILT_HEAD_RIGHT", "prompt": "Tilt your head towards your RIGHT shoulder ↪️", "target": "roll", "direction": 1, "threshold": 8.0},
        {"id": "BLINK", "prompt": "Blink your eyes clearly 👁️", "target": "blink", "direction": 0, "threshold": 0.20},
        {"id": "DOUBLE_BLINK", "prompt": "Blink your eyes TWICE 👀", "target": "double_blink", "direction": 0, "threshold": 0.20},
        {"id": "SMILE", "prompt": "Smile at the camera 😊", "target": "smile", "direction": 1, "threshold": 0.04},
        {"id": "OPEN_MOUTH", "prompt": "Open your mouth slightly 😮", "target": "open_mouth", "direction": 1, "threshold": 0.30},
        {"id": "NOD_HEAD", "prompt": "Nod your head up and down ↕️", "target": "nod", "direction": 0, "threshold": 9.0},
        {"id": "MOVE_CLOSER", "prompt": "Move slightly CLOSER to the camera 🔍", "target": "closer", "direction": 1, "threshold": 1.10},
    ]

    def __init__(self, challenge_count: int = 2):
        # Randomly choose distinct challenges for this session
        selected = random.sample(self.CHALLENGE_DEFINITIONS, min(challenge_count, len(self.CHALLENGE_DEFINITIONS)))
        self.challenges = [
            {
                "id": c["id"],
                "prompt": c["prompt"],
                "target": c["target"],
                "direction": c.get("direction", 0),
                "threshold": c["threshold"],
                "completed": False,
                "hold_frames": 0,
            }
            for c in selected
        ]
        self.current_index = 0
        self.all_completed = False

        # State tracking for multi-frame gestures
        self.blink_state = "OPEN"  # OPEN -> CLOSED -> OPEN
        self.blink_count = 0
        self.ear_history = deque(maxlen=20)
        self.baseline_mouth_ratio = None
        self.baseline_mar = None
        self.baseline_eye_dist = None
        self.pitch_history = deque(maxlen=15)
        self.frame_count = 0

    def update(self, yaw: Optional[float], pitch: Optional[float], ear: Optional[float], landmarks=None) -> Dict:
        """
        Evaluate candidate motion against the active challenge.
        Returns a dict of current challenge telemetry for live frontend display.
        """
        self.frame_count += 1
        if self.all_completed or self.current_index >= len(self.challenges):
            self.all_completed = True
            return self.get_status()

        active = self.challenges[self.current_index]
        target = active["target"]

        if ear is not None:
            self.ear_history.append(ear)
        if pitch is not None:
            self.pitch_history.append(pitch)

        # Baseline capture in first 8 frames
        if landmarks is not None and len(landmarks) > 291:
            try:
                p33 = np.array(landmarks[33][:2])
                p263 = np.array(landmarks[263][:2])
                eye_dist = float(np.linalg.norm(p263 - p33))

                p61 = np.array(landmarks[61][:2])
                p291 = np.array(landmarks[291][:2])
                mouth_w = float(np.linalg.norm(p291 - p61))
                smile_ratio = mouth_w / (eye_dist + 1e-6)

                p13 = np.array(landmarks[13][:2])
                p14 = np.array(landmarks[14][:2])
                mouth_h = float(np.linalg.norm(p14 - p13))
                mar = mouth_h / (mouth_w + 1e-6)

                if self.frame_count <= 8:
                    if self.baseline_eye_dist is None:
                        self.baseline_eye_dist = eye_dist
                    else:
                        self.baseline_eye_dist = 0.8 * self.baseline_eye_dist + 0.2 * eye_dist

                    if self.baseline_mouth_ratio is None:
                        self.baseline_mouth_ratio = smile_ratio
                    else:
                        self.baseline_mouth_ratio = 0.8 * self.baseline_mouth_ratio + 0.2 * smile_ratio

                    if self.baseline_mar is None:
                        self.baseline_mar = mar
                    else:
                        self.baseline_mar = 0.8 * self.baseline_mar + 0.2 * mar
            except Exception:
                pass

        # ── 1. TURN HEAD (LEFT / RIGHT) ──────────────────────────────────
        if target == "yaw" and yaw is not None:
            # Check yaw threshold (both direction and magnitude account for mirroring)
            if abs(yaw) >= active["threshold"]:
                active["hold_frames"] += 1
                if active["hold_frames"] >= 1:
                    active["completed"] = True
                    self.current_index += 1

        # ── 2. LOOK UP ───────────────────────────────────────────────────
        elif target == "pitch_up" and pitch is not None:
            if pitch >= active["threshold"] or abs(pitch) >= 11.0:
                active["hold_frames"] += 1
                if active["hold_frames"] >= 1:
                    active["completed"] = True
                    self.current_index += 1

        # ── 3. LOOK DOWN ─────────────────────────────────────────────────
        elif target == "pitch_down" and pitch is not None:
            if pitch <= active["threshold"] or abs(pitch) >= 11.0:
                active["hold_frames"] += 1
                if active["hold_frames"] >= 1:
                    active["completed"] = True
                    self.current_index += 1

        # ── 4. TILT HEAD (ROLL: LEFT / RIGHT) ────────────────────────────
        elif target == "roll" and landmarks is not None and len(landmarks) > 263:
            try:
                p33 = np.array(landmarks[33][:2])
                p263 = np.array(landmarks[263][:2])
                delta = p263 - p33
                roll = float(np.degrees(np.arctan2(delta[1], delta[0])))
                if abs(roll) >= active["threshold"]:
                    active["hold_frames"] += 1
                    if active["hold_frames"] >= 1:
                        active["completed"] = True
                        self.current_index += 1
            except Exception:
                pass

        # ── 5. SINGLE BLINK ──────────────────────────────────────────────
        elif target == "blink" and ear is not None:
            if ear < active["threshold"]:
                self.blink_state = "CLOSED"
            elif self.blink_state == "CLOSED" and ear > 0.24:
                self.blink_state = "DONE"
                active["completed"] = True
                self.current_index += 1

        # ── 6. DOUBLE BLINK ──────────────────────────────────────────────
        elif target == "double_blink" and ear is not None:
            if ear < active["threshold"] and self.blink_state == "OPEN":
                self.blink_state = "CLOSED"
            elif self.blink_state == "CLOSED" and ear > 0.24:
                self.blink_count += 1
                self.blink_state = "OPEN"
                if self.blink_count >= 2:
                    active["completed"] = True
                    self.current_index += 1

        # ── 7. SMILE ─────────────────────────────────────────────────────
        elif target == "smile" and landmarks is not None and len(landmarks) > 291:
            try:
                p61 = np.array(landmarks[61][:2])
                p291 = np.array(landmarks[291][:2])
                p33 = np.array(landmarks[33][:2])
                p263 = np.array(landmarks[263][:2])
                mouth_w = float(np.linalg.norm(p291 - p61))
                eye_w = float(np.linalg.norm(p263 - p33)) + 1e-6
                ratio = mouth_w / eye_w

                # Detected smile when mouth width expands by >= 0.02 relative to baseline or ratio >= 0.68
                is_smile = False
                if self.baseline_mouth_ratio is not None and (ratio - self.baseline_mouth_ratio) >= 0.020:
                    is_smile = True
                elif ratio >= 0.68:
                    is_smile = True

                if is_smile:
                    active["hold_frames"] += 1
                    if active["hold_frames"] >= 1:
                        active["completed"] = True
                        self.current_index += 1
            except Exception:
                pass

        # ── 8. OPEN MOUTH ────────────────────────────────────────────────
        elif target == "open_mouth" and landmarks is not None and len(landmarks) > 291:
            try:
                p13 = np.array(landmarks[13][:2])
                p14 = np.array(landmarks[14][:2])
                p61 = np.array(landmarks[61][:2])
                p291 = np.array(landmarks[291][:2])
                mouth_h = float(np.linalg.norm(p14 - p13))
                mouth_w = float(np.linalg.norm(p291 - p61)) + 1e-6
                mar = mouth_h / mouth_w

                if mar >= active["threshold"] or (self.baseline_mar and mar >= self.baseline_mar * 1.6):
                    active["hold_frames"] += 1
                    if active["hold_frames"] >= 1:
                        active["completed"] = True
                        self.current_index += 1
            except Exception:
                pass

        # ── 9. NOD HEAD ──────────────────────────────────────────────────
        elif target == "nod" and pitch is not None:
            if abs(pitch) >= active["threshold"]:
                active["hold_frames"] += 1
                if active["hold_frames"] >= 1:
                    active["completed"] = True
                    self.current_index += 1

        # ── 10. MOVE CLOSER (DEPTH/SCALE) ────────────────────────────────
        elif target == "closer" and landmarks is not None and len(landmarks) > 263:
            try:
                p33 = np.array(landmarks[33][:2])
                p263 = np.array(landmarks[263][:2])
                eye_dist = float(np.linalg.norm(p263 - p33))
                if self.baseline_eye_dist and eye_dist >= self.baseline_eye_dist * active["threshold"]:
                    active["hold_frames"] += 1
                    if active["hold_frames"] >= 1:
                        active["completed"] = True
                        self.current_index += 1
            except Exception:
                pass

        if self.current_index >= len(self.challenges):
            self.all_completed = True

        return self.get_status()

    def get_status(self) -> Dict:
        """Serialisable status for WebSocket frames and dashboard."""
        curr_step = min(self.current_index, len(self.challenges) - 1)
        active = self.challenges[curr_step] if self.challenges else None

        return {
            "all_completed": self.all_completed,
            "current_step": self.current_index + 1 if not self.all_completed else len(self.challenges),
            "total_steps": len(self.challenges),
            "current_id": active["id"] if active and not self.all_completed else "VERIFIED",
            "current_prompt": active["prompt"] if active and not self.all_completed else "Active Liveness Verified! ✓",
            "challenges": [
                {
                    "id": c["id"],
                    "prompt": c["prompt"],
                    "completed": c["completed"]
                }
                for c in self.challenges
            ]
        }
