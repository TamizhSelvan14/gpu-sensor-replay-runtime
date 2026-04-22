from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(slots=True)
class CameraPacket:
    timestamp_ms: int
    frame_path: Path
    sequence_id: int


@dataclass(slots=True)
class IMUPacket:
    timestamp_ms: int
    speed_mps: float
    accel_mps2: float
    yaw_rate_dps: float
    sequence_id: int


@dataclass(slots=True)
class SyncedSample:
    camera: CameraPacket
    imu: IMUPacket
    sync_gap_ms: int


@dataclass(slots=True)
class InferenceResult:
    label: str
    confidence: float
    device: str
    inference_latency_ms: float
    debug_features: dict[str, float]


@dataclass(slots=True)
class FusedEvent:
    timestamp_ms: int
    label: str
    confidence: float
    speed_mps: float
    accel_mps2: float
    risk_score: float
    risk_level: str
    sync_gap_ms: int
    inference_latency_ms: float
    debug_features: dict[str, Any]


ArrayLike = np.ndarray
