from __future__ import annotations

import csv
import math
from pathlib import Path

from PIL import Image, ImageDraw


def _label_for_index(i: int, total: int) -> str:
    phase = i / max(total, 1)
    if phase < 0.33:
        return "clear"
    if phase < 0.66:
        return "vehicle"
    return "pedestrian"


def _color_for_label(label: str) -> tuple[int, int, int]:
    return {
        "clear": (50, 80, 220),
        "vehicle": (220, 60, 60),
        "pedestrian": (60, 200, 80),
    }[label]


def generate_sample_data(
    output_dir: Path,
    frame_count: int = 120,
    frame_rate_hz: float = 10.0,
    imu_rate_hz: float = 50.0,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames_dir = output_dir / "camera_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    camera_csv = output_dir / "camera_stream.csv"
    imu_csv = output_dir / "imu_stream.csv"

    frame_interval_ms = int(1000 / frame_rate_hz)
    imu_interval_ms = int(1000 / imu_rate_hz)
    total_duration_ms = (frame_count - 1) * frame_interval_ms
    imu_count = int(total_duration_ms / imu_interval_ms) + 1

    with camera_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "timestamp_ms", "frame_path", "expected_label"])
        for i in range(frame_count):
            ts = i * frame_interval_ms
            label = _label_for_index(i, frame_count)
            frame_path = frames_dir / f"frame_{i:04d}.png"
            _write_frame(frame_path, i, frame_count, label)
            writer.writerow([i, ts, frame_path.name, label])

    with imu_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["sequence_id", "timestamp_ms", "speed_mps", "accel_mps2", "yaw_rate_dps"])
        for i in range(imu_count):
            ts = i * imu_interval_ms
            phase = ts / max(total_duration_ms, 1)
            speed = 4.0 + 8.0 * phase + 1.2 * math.sin(phase * 4 * math.pi)
            accel = 0.2 + 1.6 * math.sin(phase * 6 * math.pi)
            yaw = 2.0 * math.cos(phase * 3 * math.pi)
            writer.writerow([i, ts, round(speed, 3), round(accel, 3), round(yaw, 3)])


def _write_frame(frame_path: Path, index: int, total: int, label: str) -> None:
    width, height = 320, 240
    image = Image.new("RGB", (width, height), color=(18, 18, 24))
    draw = ImageDraw.Draw(image)

    # lane / horizon lines
    draw.rectangle((0, 150, width, height), fill=(28, 28, 32))
    draw.line((0, 150, width, 150), fill=(90, 90, 90), width=2)
    draw.line((width // 2, 150, width // 2, height), fill=(140, 140, 140), width=3)

    color = _color_for_label(label)
    x = int(40 + (index / max(total - 1, 1)) * 220)
    y = 120 if label == "clear" else 110
    size = 28 if label == "pedestrian" else 42
    draw.rectangle((x, y, x + size, y + size), fill=color)

    # motion trail for later phases
    if label != "clear":
        for t in range(1, 4):
            alpha_x = x - 8 * t
            alpha_y = y + t
            shade = tuple(max(0, c - 40 * t) for c in color)
            draw.rectangle((alpha_x, alpha_y, alpha_x + size, alpha_y + size), outline=shade)

    image.save(frame_path)
