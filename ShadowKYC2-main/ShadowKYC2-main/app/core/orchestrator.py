"""
SessionOrchestrator – 7-Layer Pipeline
Manages the full video processing workflow for SHADOW KYC.
"""
import uuid
import time
from typing import List, Dict, Optional
from dataclasses import asdict

from app.modules.frame_sampler import FrameSampler
from app.modules.face_detector import FaceDetector
from app.modules.evidence_manager import EvidenceManager
from app.modules.report_generator import ReportGenerator

# ── 7 Integrity Layers ────────────────────────────────────────────────────────
from app.modules.layer_1_temporal_liveness import TemporalLivenessEngine
from app.modules.layer_2_face_match        import FaceMatchMonitor
from app.modules.layer_3_texture_artifact  import TextureArtifactDetector
from app.modules.layer_4_geometry          import GeometryIntegrityChecker
from app.modules.layer_5_lipsync           import LipSyncValidator
from app.modules.layer_6_environment       import EnvironmentIntegrityChecker
from app.modules.layer_7_meta_fusion       import MetaFusionEngine
from app.modules.challenge_manager import ChallengeManager
from app.modules.deepfake_detector import DeepfakeDetector

from app.core.config import RISK_HIGH_THRESHOLD, RISK_LOW_THRESHOLD
from app.core.types  import FrameData, SessionReport


class SessionOrchestrator:
    """
    Core orchestrator that manages the 7-layer video processing pipeline,
    frame extraction, layer execution, and result aggregation.
    """

    def __init__(self):
        self.session_id = str(uuid.uuid4())

        # ── Pre-processing ────────────────────────────────────────────────
        self.frame_sampler  = FrameSampler()
        self.face_detector  = FaceDetector()

        # ── 7 Layers (share the face detector for landmarks) ──────────────
        self.l1 = TemporalLivenessEngine(face_detector=self.face_detector)   # L1 – Temporal Liveness
        self.l2 = FaceMatchMonitor()            # L2 – Face Match
        self.l3 = TextureArtifactDetector()     # L3 – Texture/Frequency
        self.l4 = GeometryIntegrityChecker(face_detector=self.face_detector) # L4 – Geometry/Depth
        self.l5 = LipSyncValidator(face_detector=self.face_detector)         # L5 – Lip-Sync
        self.l6 = EnvironmentIntegrityChecker() # L6 – Environment
        self.l7 = MetaFusionEngine()            # L7 – Meta Fusion
        self.challenge_manager = ChallengeManager(challenge_count=2)
        self.deepfake_detector = DeepfakeDetector()

        # Startup diagnostics for every detector
        self.detector_diagnostics = self.face_detector.diagnostics()
        print(f"[Orchestrator] Detector diagnostics: {self.detector_diagnostics}")

        # ── Output ────────────────────────────────────────────────────────
        self.evidence_manager  = EvidenceManager(self.session_id)
        self.report_generator  = ReportGenerator(self.session_id)

        # ── Session accumulators ──────────────────────────────────────────
        self.frames_analyzed       = 0
        self.flagged_frames_count  = 0
        self.accumulated_risk      = 0.0
        self.reason_codes          = set()
        self.suspicious_timestamps = []
        self.evidence_paths        = []

        # Per-layer score sums for session averages
        self.scores_sum = {
            "l1_liveness":    0.0,
            "l2_face_match":  0.0,
            "l3_texture":     0.0,
            "l4_geometry":    0.0,
            "l5_lipsync":     0.0,
            "l6_environment": 0.0,
        }

    # ── Public API ────────────────────────────────────────────────────────────
    def start_session(self, video_path: str) -> SessionReport:
        """Process a video file and return the full session report."""
        print(f"[Orchestrator] Starting session {self.session_id} -> {video_path}")
        try:
            frames_iter = self.frame_sampler.extract_frames(video_path)
        except Exception as e:
            print(f"[Orchestrator] Frame sampler error: {e}")
            return self._error_report(str(e))

        for frame, timestamp in frames_iter:
            self.process_single_frame(frame, timestamp)

        return self._finalize_session()

    def process_single_frame(self, frame, timestamp: float) -> Dict:
        """
        Process a single frame through all 7 layers.
        Returns a serialisable dict for live WebSocket streaming.
        """
        prev_evidence_count = len(self.evidence_paths)
        frame_data = self._run_pipeline(frame, timestamp)
        new_evidence_path = (
            self.evidence_paths[-1]
            if len(self.evidence_paths) > prev_evidence_count
            else None
        )

        # L7 fusion result for this frame
        fusion = self.l7.compute_risk(frame_data)

        return {
            "timestamp":      timestamp,
            "challenge":      self.challenge_manager.get_status(),
            # Layer goodness scores (1.0 = clean)
            "liveness":       frame_data.liveness_score,
            "face_match":     frame_data.metadata.get("l2_match_score"),
            "texture":        frame_data.artifact_score,
            "geometry":       frame_data.metadata.get("l4_geometry_score"),
            "lipsync":        frame_data.metadata.get("l5_lipsync_score"),
            "environment":    frame_data.metadata.get("l6_env_score"),
            # Legacy aliases for frontend compatibility
            "quality":        frame_data.liveness_score,
            "artifact":       frame_data.artifact_score,
            "deepfake":       frame_data.metadata.get("deepfake_score", 1.0 - fusion["risk_score"]),
            "deepfake_probability": frame_data.metadata.get("deepfake_probability", 0.0),
            "deepfake_detected": frame_data.metadata.get("deepfake_detected", False),
            "deepfake_flags":    frame_data.metadata.get("deepfake_flags", []),
            # Fusion output
            "risk":           fusion["risk_score"],
            "risk_pct":       fusion["risk_pct"],
            "classification": fusion["classification"],
            "recommendation": fusion["recommendation"],
            "flags":          frame_data.flags,
            "top_reasons":    fusion["top_reasons"],
            "layer_scores":   fusion["layer_scores"],
            "layer_status":   fusion.get("layer_status", {}),
            "evidence_path":  new_evidence_path,
            # Detector availability (startup diagnostics snapshot)
            "detectors":      {
                "face_detection": self.face_detector.haar_available,
                "face_landmarks": self.face_detector.use_mediapipe,
            },
        }

    def get_current_stats(self) -> Dict:
        """Rolling session statistics for live dashboard."""
        n = max(self.frames_analyzed, 1)
        avg_risk = self.accumulated_risk / n

        # If active 3D liveness challenge has been verified, reflect low risk immediately
        has_deepfake = any(
            f in self.reason_codes
            for f in ("DEEPFAKE_SYNTHETIC_FACE_DETECTED", "DEEPFAKE_BLENDING_SEAM_DETECTED", "DEEPFAKE_GAN_GRID_ARTIFACT")
        )
        if has_deepfake:
            avg_risk = max(avg_risk, 0.92)
            classification = "HIGH_RISK"
            recommendation = "REJECT – Synthetic deepfake, face-swap blending seam, or generative AI manipulation detected."
        elif self.challenge_manager.all_completed:
            avg_risk = min(avg_risk, 0.08)

        classification = "LOW_RISK"
        if avg_risk > RISK_HIGH_THRESHOLD:
            classification = "HIGH_RISK"
        elif avg_risk > RISK_LOW_THRESHOLD:
            classification = "MEDIUM_RISK"

        avg_scores = {k: round(v / n, 4) for k, v in self.scores_sum.items()}

        return {
            "session_id":    self.session_id,
            "active_challenge_completed": self.challenge_manager.all_completed,
            "challenge":      self.challenge_manager.get_status(),
            "avg_risk":      round(avg_risk, 4),
            "risk_pct":      int(avg_risk * 100),
            "classification": classification,
            "frames_count":  self.frames_analyzed,
            "flagged_count": self.flagged_frames_count,
            "layer_averages": avg_scores,
            # Legacy key for frontend radar chart
            "temporal":      avg_scores.get("l1_liveness", 0.0),
        }

    def finalize_live_session(self) -> SessionReport:
        return self._finalize_session()

    @staticmethod
    def _goodness(score) -> float:
        """Goodness score (1.0 = clean). Missing/unavailable -> neutral 0.50
        (cannot be assumed 100% clean if not detected)."""
        return float(score) if score is not None else 1.0

    # ── Internal pipeline ─────────────────────────────────────────────────────
    def _run_pipeline(self, frame, timestamp: float) -> FrameData:
        frame_data = FrameData(image=frame, timestamp_sec=timestamp)

        # Face detection (provides ROI for texture/artifact layers)
        roi, bbox = self.face_detector.process_frame(frame)
        frame_data.face_roi = roi

        # ── L1: Temporal Liveness ─────────────────────────────────────────
        self.l1.process(frame_data)

        # ── Active Liveness Challenge Evaluation ─────────────────────────
        yaw = frame_data.metadata.get("l1_yaw")
        pitch = frame_data.metadata.get("l1_pitch")
        ear = frame_data.metadata.get("l1_ear")
        landmarks = frame_data.metadata.get("landmarks")
        challenge_status = self.challenge_manager.update(yaw, pitch, ear, landmarks)
        frame_data.metadata["active_challenge_completed"] = self.challenge_manager.all_completed
        frame_data.metadata["active_challenge_status"] = challenge_status

        # ── L2: Face Match Monitoring ─────────────────────────────────────
        self.l2.process(frame_data)

        # ── L3: Texture / Frequency Artifact ─────────────────────────────
        self.l3.process(frame_data)

        # ── L4: Geometry / Pseudo-Depth ──────────────────────────────────
        self.l4.process(frame_data)

        # ── L5: Lip-Sync Integrity ────────────────────────────────────────
        self.l5.process(frame_data)

        # ── L6: Environment & Background ─────────────────────────────────
        self.l6.process(frame_data)

        # ── Deepfake Forensic Analysis ──────────────────────────────────
        self.deepfake_detector.process(frame_data)

        # ── L7: Meta Fusion (compute risk for accumulation) ───────────────
        fusion = self.l7.compute_risk(frame_data)
        final_risk = fusion["risk_score"]

        # ── Accumulate stats ──────────────────────────────────────────────
        # A missing/unavailable layer (score is None) contributes a neutral
        # goodness of 1.0 so it is never mistaken for evidence of fraud.
        self.frames_analyzed += 1
        self.accumulated_risk += final_risk

        meta = frame_data.metadata
        self.scores_sum["l1_liveness"] += self._goodness(frame_data.liveness_score)
        self.scores_sum["l2_face_match"] += self._goodness(meta.get("l2_match_score"))
        self.scores_sum["l3_texture"] += self._goodness(frame_data.artifact_score)
        self.scores_sum["l4_geometry"] += self._goodness(meta.get("l4_geometry_score"))
        self.scores_sum["l5_lipsync"] += self._goodness(meta.get("l5_lipsync_score"))
        self.scores_sum["l6_environment"] += self._goodness(meta.get("l6_env_score"))

        # ── Evidence capture ──────────────────────────────────────────────
        if frame_data.flags or final_risk > 0.30:
            path = self.evidence_manager.process(frame_data)
            if path:
                self.evidence_paths.append(path)
            self.flagged_frames_count += 1
            self.suspicious_timestamps.append(timestamp)
            self.reason_codes.update(frame_data.flags)

        return frame_data

    def _finalize_session(self) -> SessionReport:
        n = max(self.frames_analyzed, 1)
        avg_risk = self.accumulated_risk / n

        classification = "LOW_RISK"
        if avg_risk > RISK_HIGH_THRESHOLD:
            classification = "HIGH_RISK"
        elif avg_risk > RISK_LOW_THRESHOLD:
            classification = "MEDIUM_RISK"

        # Finalization aligned with Active 3D Liveness and Passive Neural Anti-Spoof
        has_deepfake = any(
            f in self.reason_codes
            for f in ("DEEPFAKE_SYNTHETIC_FACE_DETECTED", "DEEPFAKE_BLENDING_SEAM_DETECTED", "DEEPFAKE_GAN_GRID_ARTIFACT")
        )
        if has_deepfake:
            avg_risk = max(avg_risk, 0.92)
            classification = "HIGH_RISK"
            recommendation = "REJECT – Synthetic deepfake, face-swap blending seam, or generative AI manipulation detected."
        elif self.challenge_manager.all_completed:
            avg_risk = min(avg_risk, 0.08)
            classification = "LOW_RISK"
            recommendation = "APPROVE – Active 3D liveness verified and biometrics authenticated."
        elif self.frames_analyzed >= 15:
            self.reason_codes.add("ACTIVE_LIVENESS_INCOMPLETE")
            avg_risk = max(avg_risk, 0.70)
            classification = "HIGH_RISK"
            recommendation = "REJECT – Active liveness challenges were not satisfied (photo or unverified feed)."
        elif avg_risk >= 0.50 or self.flagged_frames_count >= 5:
            classification = "HIGH_RISK"
            recommendation = "REJECT – Presentation attack or synthetic video detected."
        elif avg_risk >= 0.30 or self.flagged_frames_count >= 2:
            classification = "MEDIUM_RISK"
            recommendation = "REVIEW – Manual officer verification recommended."
        else:
            recommendation = "APPROVE – Session appears authentic."

        avg_scores = {k: round(v / n, 4) for k, v in self.scores_sum.items()}

        report = SessionReport(
            session_id=self.session_id,
            risk_score=round(avg_risk, 4),
            risk_pct=int(avg_risk * 100),
            classification=classification,
            recommendation=recommendation,
            layer_scores=avg_scores,
            reason_codes=sorted(self.reason_codes),
            top_suspicious_timestamps=sorted(self.suspicious_timestamps)[:10],
            flagged_frame_count=self.flagged_frames_count,
            evidence_frames=self.evidence_paths,
            deepfake_probability=None,
            replay_score=avg_scores.get("l1_liveness", 0.0),
            liveness_score=avg_scores.get("l1_liveness", 0.0),
        )

        output_path = self.report_generator.generate(report)
        print(f"[Orchestrator] Session finalised. Report -> {output_path}")
        return report

    def _error_report(self, error_msg: str) -> SessionReport:
        return SessionReport(
            session_id=self.session_id,
            risk_score=0.0,
            risk_pct=0,
            classification="ERROR",
            recommendation=f"ERROR: {error_msg}",
            layer_scores={},
            reason_codes=[f"ERROR: {error_msg}"],
            top_suspicious_timestamps=[],
            flagged_frame_count=0,
            evidence_frames=[],
            deepfake_probability=0.0,
            replay_score=0.0,
            liveness_score=0.0,
        )
