from pathlib import Path

from src.gpu_sensor_runtime.replay.synchronizer import SyncCoordinator
from src.gpu_sensor_runtime.types import CameraPacket, IMUPacket


def test_camera_imu_sync_within_tolerance() -> None:
    sync = SyncCoordinator(tolerance_ms=20)
    sync.ingest_imu(IMUPacket(timestamp_ms=100, speed_mps=3.0, accel_mps2=0.1, yaw_rate_dps=0.0, sequence_id=1))
    camera = CameraPacket(timestamp_ms=112, frame_path=Path("frame.png"), sequence_id=1)
    aligned = sync.try_sync_camera(camera)
    assert aligned is not None
    assert aligned.sync_gap_ms == 12
