from __future__ import annotations

from src.gpu_sensor_runtime.types import FusedEvent, InferenceResult, SyncedSample


class EventFuser:
    def fuse(self, sample: SyncedSample, result: InferenceResult) -> FusedEvent:
        speed = sample.imu.speed_mps
        accel = abs(sample.imu.accel_mps2)

        base = result.confidence
        if result.label == "vehicle":
            risk_score = min(1.0, 0.55 * base + 0.03 * speed + 0.05 * accel)
        elif result.label == "pedestrian":
            risk_score = min(1.0, 0.45 * base + 0.015 * speed + 0.08 * accel)
        else:
            risk_score = min(1.0, 0.2 * base + 0.01 * accel)

        if risk_score >= 0.72:
            risk_level = "HIGH"
        elif risk_score >= 0.45:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"

        return FusedEvent(
            timestamp_ms=sample.camera.timestamp_ms,
            label=result.label,
            confidence=result.confidence,
            speed_mps=sample.imu.speed_mps,
            accel_mps2=sample.imu.accel_mps2,
            risk_score=round(risk_score, 5),
            risk_level=risk_level,
            sync_gap_ms=sample.sync_gap_ms,
            inference_latency_ms=round(result.inference_latency_ms, 5),
            debug_features=result.debug_features,
        )
