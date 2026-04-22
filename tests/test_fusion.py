from src.gpu_sensor_runtime.fusion.fuser import EventFuser
from src.gpu_sensor_runtime.types import CameraPacket, IMUPacket, InferenceResult, SyncedSample
from pathlib import Path


def test_high_risk_vehicle_event() -> None:
    sample = SyncedSample(
        camera=CameraPacket(timestamp_ms=1000, frame_path=Path("frame.png"), sequence_id=1),
        imu=IMUPacket(timestamp_ms=1003, speed_mps=14.0, accel_mps2=1.8, yaw_rate_dps=0.3, sequence_id=1),
        sync_gap_ms=3,
    )
    result = InferenceResult(
        label="vehicle",
        confidence=0.9,
        device="cpu",
        inference_latency_ms=1.2,
        debug_features={},
    )
    event = EventFuser().fuse(sample, result)
    assert event.risk_level == "HIGH"
