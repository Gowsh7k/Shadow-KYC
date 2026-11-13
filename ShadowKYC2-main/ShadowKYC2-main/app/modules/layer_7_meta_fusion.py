"""
Layer 7 – Meta Fusion + Explainable Final Decision Engine
Purpose: Synthesize L1–L6 signals, execute hard security vetoes, and deliver strict fraud decisions.
Tech Stack: Python + NumPy + Rule Engine
Output: Final risk score (0–100) + veto tags + flagged frames + recommendation
"""
import numpy as np
from app.core.types import FrameData
from app.core import config


LAYER_WEIGHTS = {
    "l1_liveness": 0.25,
    "l2_face_match": 0.20,
    "l3_texture": 0.25,  # Boosted for MiniFASNet neural anti-spoofing
    "l4_geometry": 0.10,
    "l5_lipsync": 0.10,
    "l6_environment": 0.10,
}

# ── Critical Security Veto Rules ─────────────────────────────────────────────
# In banking security, these triggers represent definitive presentation or identity attacks.
# When any veto flag is triggered, risk is escalated directly to reject thresholds.
CRITICAL_SPOOF_VETOES = {
    "L3_SCREEN_REPLAY_DETECTED": 0.95,     # Detected screen/monitor/phone display
    "L3_PRINT_PHOTO_DETECTED": 0.95,      # Detected printed paper photo attack
    "FROZEN_FRAME_REPLAY": 0.92,          # Static frozen photo or replay attack
    "L1_STATIC_PHOTO_DETECTED": 0.92,     # Zero blink + zero head micro-motion across frames
    "L2_FACE_SWAP_DETECTED": 0.95,        # Candidate face swapped with another identity
    "L3_LOW_HF_ENERGY_AI_FACE": 0.85,     # AI synthetic face missing biological detail
    "L3_GAN_GRID_ARTIFACT": 0.85,         # Generative AI checkerboard / grid artifacts
    "CONTRADICTION_SWAP_AND_REPLAY": 0.95,# Simultaneous identity swap and video replay
    # ── Deepfake Forensic Vetoes ──────────────────────────────────────────────
    "DEEPFAKE_SYNTHETIC_FACE_DETECTED": 0.95,
    "DEEPFAKE_BLENDING_SEAM_DETECTED": 0.92,
    "DEEPFAKE_CORNEAL_SPECULAR_MISMATCH": 0.88,
    "DEEPFAKE_GAN_GRID_ARTIFACT": 0.90,
    "DEEPFAKE_TEMPORAL_MASK_FLUTTER": 0.88,
}

CONTRADICTION_RULES = [
    ("L1_UNNATURAL_HEAD_STABILITY", "L6_BACKGROUND_MOTION_DETECTED", 0.15, "CONTRADICTION_STATIC_FACE_MOVING_BG"),
    ("L2_FACE_SWAP_DETECTED", "FROZEN_FRAME_REPLAY", 0.20, "CONTRADICTION_SWAP_AND_REPLAY"),
    ("L3_OVER_SMOOTH_FACE", "L4_EYE_WIDTH_DISTORTION", 0.15, "CONTRADICTION_SMOOTH_WARPED"),
    ("L5_MOUTH_FROZEN_NO_SPEECH", "L6_BACKGROUND_MOTION_DETECTED", 0.15, "CONTRADICTION_SILENT_FACE_ACTIVE_BG"),
]

RISK_HIGH = 0.60
RISK_MEDIUM = 0.30


def _layer_risk_contribution(score):
    if score is None:
        return 0.40  # Missing/inconclusive biometrics are mildly suspicious (not 0.0)
    return float(np.clip(1.0 - float(score), 0.0, 1.0))


class MetaFusionEngine:
    def __init__(self):
        print("L7-MetaFusion: Hard-veto security fusion engine ready")

    def compute_risk(self, frame_data: FrameData) -> dict:
        meta = frame_data.metadata
        flags = set(frame_data.flags)

        layer_goodness = {
            "l1_liveness": frame_data.liveness_score,
            "l2_face_match": meta.get("l2_match_score"),
            "l3_texture": frame_data.artifact_score,
            "l4_geometry": meta.get("l4_geometry_score"),
            "l5_lipsync": meta.get("l5_lipsync_score"),
            "l6_environment": meta.get("l6_env_score"),
        }

        layer_status = {
            "l1_liveness": meta.get("l1_status", config.LAYER_STATUS_UNAVAILABLE),
            "l2_face_match": meta.get("l2_status", config.LAYER_STATUS_UNAVAILABLE),
            "l3_texture": meta.get("l3_status", config.LAYER_STATUS_UNAVAILABLE),
            "l4_geometry": meta.get("l4_status", config.LAYER_STATUS_UNAVAILABLE),
            "l5_lipsync": meta.get("l5_status", config.LAYER_STATUS_UNAVAILABLE),
            "l6_environment": meta.get("l6_status", config.LAYER_STATUS_UNAVAILABLE),
        }

        # Dynamic normalization over active layers:
        # Inactive/unavailable layers (e.g. L5 when silent, L2 when no doc uploaded)
        # do NOT artificially inflate the risk of an authentic user.
        layer_scores = {}
        active_weights = 0.0
        weighted_risk_sum = 0.0

        for k, weight in LAYER_WEIGHTS.items():
            val = layer_goodness[k]
            if val is not None:
                risk_contrib = float(np.clip(1.0 - float(val), 0.0, 1.0))
                layer_scores[k] = round(risk_contrib, 4)
                weighted_risk_sum += risk_contrib * weight
                active_weights += weight
            else:
                layer_scores[k] = 0.0

        base_risk = (weighted_risk_sum / active_weights) if active_weights > 0 else 0.0

        contradiction_reasons = []
        contradiction_penalty = 0.0
        for flag_a, flag_b, penalty, tag in CONTRADICTION_RULES:
            if flag_a in flags and flag_b in flags:
                contradiction_penalty += penalty
                contradiction_reasons.append(tag)

        # ── Critical Security Veto Enforcement ────────────────────────────
        max_veto_risk = 0.0
        veto_tags = []
        is_active_live = meta.get("active_challenge_completed", False)

        for flag in flags:
            if flag in CRITICAL_SPOOF_VETOES:
                veto_val = CRITICAL_SPOOF_VETOES[flag]
                # If active 3D liveness has been mathematically confirmed by interactive challenge completion,
                # mitigate false passive screen replay alarms from standard webcams
                if is_active_live and flag in ("L3_SCREEN_REPLAY_DETECTED", "L3_PRINT_PHOTO_DETECTED"):
                    veto_val = 0.15
                if veto_val > max_veto_risk:
                    max_veto_risk = veto_val
                veto_tags.append(flag)

        if "ACTIVE_LIVENESS_INCOMPLETE" in flags or "ACTIVE_LIVENESS_FAILED" in flags:
            max_veto_risk = max(max_veto_risk, 0.75)
            veto_tags.append("ACTIVE_LIVENESS_INCOMPLETE")

        raw_risk = max(base_risk + contradiction_penalty, max_veto_risk)
        if is_active_live and max_veto_risk < 0.60:
            final_risk = float(np.clip(min(raw_risk, 0.15), 0.0, 1.0))
        else:
            final_risk = float(np.clip(raw_risk, 0.0, 1.0))

        # ── Strict Decision Thresholds ────────────────────────────────────
        if max_veto_risk >= 0.70 or final_risk >= RISK_HIGH:
            classification = "HIGH_RISK"
            recommendation = "REJECT – Presentation attack, photo spoofing, or synthetic video detected."
        elif final_risk >= RISK_MEDIUM:
            classification = "MEDIUM_RISK"
            recommendation = "REVIEW – Manual officer verification required."
        else:
            classification = "LOW_RISK"
            recommendation = "APPROVE – Session appears authentic."

        top_reasons = veto_tags + list(flags) + contradiction_reasons
        seen = set()
        deduped = []
        for r in top_reasons:
            if r not in seen:
                seen.add(r)
                deduped.append(r)

        return {
            "risk_score": round(final_risk, 4),
            "risk_pct": int(final_risk * 100),
            "classification": classification,
            "recommendation": recommendation,
            "top_reasons": deduped[:6],
            "layer_scores": layer_scores,
            "layer_status": layer_status,
            "contradiction_penalty": round(contradiction_penalty, 4),
            "max_veto_risk": round(max_veto_risk, 4),
        }
