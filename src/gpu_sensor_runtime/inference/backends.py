from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from PIL import Image

from src.gpu_sensor_runtime.types import InferenceResult, SyncedSample

try:
    import torch
except Exception:  # pragma: no cover
    torch = None


class InferenceBackend(Protocol):
    device: str

    def infer(self, sample: SyncedSample) -> InferenceResult:
        ...


@dataclass(slots=True)
class TorchHeuristicInferenceBackend:
    device: str

    @classmethod
    def create(cls, device: str = "auto") -> "TorchHeuristicInferenceBackend":
        if torch is None:
            return cls(device="cpu")
        if device == "cuda" and torch.cuda.is_available():
            return cls(device="cuda")
        if device == "auto" and torch.cuda.is_available():
            return cls(device="cuda")
        # Use the lightweight NumPy path on CPU to keep replay fast and deterministic.
        return cls(device="cpu")

    def infer(self, sample: SyncedSample) -> InferenceResult:
        start = time.perf_counter()
        image = Image.open(sample.camera.frame_path).convert("RGB")
        arr = np.asarray(image).astype(np.float32) / 255.0

        if torch is not None and self.device == "cuda":
            tensor = torch.from_numpy(arr).to(self.device)
            h, w, _ = tensor.shape
            crop = tensor[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4, :]
            channel_means = crop.mean(dim=(0, 1))
            brightness = crop.mean()
            motion_proxy = torch.tensor([
                sample.imu.speed_mps,
                abs(sample.imu.accel_mps2),
                abs(sample.imu.yaw_rate_dps),
            ], device=self.device, dtype=torch.float32)
            speed_factor = torch.sigmoid((motion_proxy[0] - 7.0) / 2.0)
            accel_factor = torch.sigmoid((motion_proxy[1] - 0.8) / 0.5)
            logits = channel_means.clone()
            logits[0] = logits[0] + 0.30 * speed_factor + 0.15 * accel_factor
            logits[1] = logits[1] + 0.12 * accel_factor
            logits[2] = logits[2] + 0.08 * (1.0 - speed_factor)
            probs = torch.softmax(logits * 4.0, dim=0)
            idx = int(torch.argmax(probs).item())
            conf = float(probs[idx].item())
            channel_vals = channel_means.detach().cpu().numpy().tolist()
            brightness_val = float(brightness.item())
        else:
            crop = arr[arr.shape[0] // 4 : 3 * arr.shape[0] // 4, arr.shape[1] // 4 : 3 * arr.shape[1] // 4, :]
            channel_means = crop.mean(axis=(0, 1))
            brightness_val = float(crop.mean())
            logits = channel_means.copy()
            speed_factor = 1 / (1 + np.exp(-(sample.imu.speed_mps - 7.0) / 2.0))
            accel_factor = 1 / (1 + np.exp(-(abs(sample.imu.accel_mps2) - 0.8) / 0.5))
            logits[0] += 0.30 * speed_factor + 0.15 * accel_factor
            logits[1] += 0.12 * accel_factor
            logits[2] += 0.08 * (1.0 - speed_factor)
            exps = np.exp(logits * 4.0)
            probs = exps / exps.sum()
            idx = int(np.argmax(probs))
            conf = float(probs[idx])
            channel_vals = channel_means.tolist()

        label = ["vehicle", "pedestrian", "clear"][idx]
        latency_ms = (time.perf_counter() - start) * 1000
        return InferenceResult(
            label=label,
            confidence=conf,
            device=self.device,
            inference_latency_ms=latency_ms,
            debug_features={
                "red_mean": round(channel_vals[0], 5),
                "green_mean": round(channel_vals[1], 5),
                "blue_mean": round(channel_vals[2], 5),
                "brightness": round(brightness_val, 5),
            },
        )
