from __future__ import annotations

import queue
from collections import deque
from pathlib import Path
from typing import Any

from src.gpu_sensor_runtime.fusion.fuser import EventFuser
from src.gpu_sensor_runtime.inference.backends import TorchHeuristicInferenceBackend
from src.gpu_sensor_runtime.observability.metrics import MetricsCollector, write_summary, write_timeline
from src.gpu_sensor_runtime.replay.loaders import ReplayThread, StreamSentinel, load_camera_packets, load_imu_packets
from src.gpu_sensor_runtime.replay.synchronizer import SyncCoordinator
from src.gpu_sensor_runtime.types import CameraPacket, FusedEvent, IMUPacket


def run_pipeline(
    data_dir: Path,
    output_dir: Path,
    device: str = "auto",
    sync_tolerance_ms: float = 30.0,
    replay_speed: float = 20.0,
) -> dict[str, Any]:
    camera_packets = load_camera_packets(data_dir)
    imu_packets = load_imu_packets(data_dir)

    camera_queue: queue.Queue = queue.Queue()
    imu_queue: queue.Queue = queue.Queue()

    camera_thread = ReplayThread(camera_packets, camera_queue, "camera", replay_speed)
    imu_thread = ReplayThread(imu_packets, imu_queue, "imu", replay_speed)

    camera_thread.start()
    imu_thread.start()

    coordinator = SyncCoordinator(tolerance_ms=int(sync_tolerance_ms))
    backend = TorchHeuristicInferenceBackend.create(device=device)
    fuser = EventFuser()
    metrics = MetricsCollector()
    events: list[FusedEvent] = []

    imu_done = False
    camera_done = False
    pending_cameras = deque()

    def _process_pending() -> None:
        nonlocal pending_cameras
        keep = deque()
        while pending_cameras:
            cam = pending_cameras.popleft()
            synced = coordinator.try_sync_camera(cam)
            if synced is None:
                keep.append(cam)
                continue
            inference = backend.infer(synced)
            event = fuser.fuse(synced, inference)
            metrics.record(event, backend.device)
            events.append(event)
        pending_cameras = keep

    while not (imu_done and camera_done):
        try:
            imu_item = imu_queue.get(timeout=0.01)
            if isinstance(imu_item, StreamSentinel):
                imu_done = True
            elif isinstance(imu_item, IMUPacket):
                coordinator.ingest_imu(imu_item)
                _process_pending()
        except queue.Empty:
            pass

        try:
            cam_item = camera_queue.get(timeout=0.01)
            if isinstance(cam_item, StreamSentinel):
                camera_done = True
            elif isinstance(cam_item, CameraPacket):
                pending_cameras.append(cam_item)
                _process_pending()
        except queue.Empty:
            pass

    _process_pending()

    camera_thread.join(timeout=1.0)
    imu_thread.join(timeout=1.0)

    summary = metrics.summary()
    write_timeline(events, output_dir)
    write_summary(summary, output_dir)
    return summary
