from __future__ import annotations

import csv
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Generic, TypeVar

from src.gpu_sensor_runtime.types import CameraPacket, IMUPacket

T = TypeVar("T")


@dataclass(slots=True)
class StreamSentinel:
    stream_name: str


class ReplayThread(threading.Thread, Generic[T]):
    def __init__(self, packets: list[T], out_queue: queue.Queue, stream_name: str, replay_speed: float) -> None:
        super().__init__(daemon=True)
        self.packets = packets
        self.out_queue = out_queue
        self.stream_name = stream_name
        self.replay_speed = max(replay_speed, 1.0)

    def run(self) -> None:
        previous_ts = None
        for packet in self.packets:
            timestamp_ms = getattr(packet, "timestamp_ms")
            if previous_ts is not None:
                delta_s = ((timestamp_ms - previous_ts) / 1000.0) / self.replay_speed
                if delta_s > 0:
                    time.sleep(delta_s)
            self.out_queue.put(packet)
            previous_ts = timestamp_ms
        self.out_queue.put(StreamSentinel(self.stream_name))



def load_camera_packets(data_dir: Path) -> list[CameraPacket]:
    packets: list[CameraPacket] = []
    with (data_dir / "camera_stream.csv").open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            packets.append(
                CameraPacket(
                    timestamp_ms=int(row["timestamp_ms"]),
                    frame_path=data_dir / "camera_frames" / row["frame_path"],
                    sequence_id=int(row["sequence_id"]),
                )
            )
    return packets



def load_imu_packets(data_dir: Path) -> list[IMUPacket]:
    packets: list[IMUPacket] = []
    with (data_dir / "imu_stream.csv").open() as f:
        reader = csv.DictReader(f)
        for row in reader:
            packets.append(
                IMUPacket(
                    timestamp_ms=int(row["timestamp_ms"]),
                    speed_mps=float(row["speed_mps"]),
                    accel_mps2=float(row["accel_mps2"]),
                    yaw_rate_dps=float(row["yaw_rate_dps"]),
                    sequence_id=int(row["sequence_id"]),
                )
            )
    return packets
