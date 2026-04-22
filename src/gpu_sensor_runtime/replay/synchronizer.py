from __future__ import annotations

from collections import deque
from typing import Deque

from src.gpu_sensor_runtime.types import CameraPacket, IMUPacket, SyncedSample


class SyncCoordinator:
    def __init__(self, tolerance_ms: int = 30) -> None:
        self.tolerance_ms = tolerance_ms
        self.imu_buffer: Deque[IMUPacket] = deque(maxlen=256)

    def ingest_imu(self, packet: IMUPacket) -> None:
        self.imu_buffer.append(packet)

    def try_sync_camera(self, packet: CameraPacket) -> SyncedSample | None:
        if not self.imu_buffer:
            return None

        best = min(self.imu_buffer, key=lambda imu: abs(imu.timestamp_ms - packet.timestamp_ms))
        gap = abs(best.timestamp_ms - packet.timestamp_ms)
        if gap > self.tolerance_ms:
            return None
        return SyncedSample(camera=packet, imu=best, sync_gap_ms=gap)
