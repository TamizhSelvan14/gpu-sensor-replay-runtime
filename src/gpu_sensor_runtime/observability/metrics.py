from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from src.gpu_sensor_runtime.types import FusedEvent


@dataclass
class MetricsCollector:
    sync_gaps_ms: list[int] = field(default_factory=list)
    inference_latencies_ms: list[float] = field(default_factory=list)
    risk_levels: list[str] = field(default_factory=list)
    device: str = "cpu"

    def record(self, event: FusedEvent, device: str) -> None:
        self.sync_gaps_ms.append(event.sync_gap_ms)
        self.inference_latencies_ms.append(event.inference_latency_ms)
        self.risk_levels.append(event.risk_level)
        self.device = device

    def _p95(self, values: list[float]) -> float:
        if not values:
            return 0.0
        sorted_vals = sorted(values)
        index = min(len(sorted_vals) - 1, int(0.95 * (len(sorted_vals) - 1)))
        return round(sorted_vals[index], 5)

    def summary(self) -> dict[str, Any]:
        high_count = sum(1 for level in self.risk_levels if level == "HIGH")
        med_count = sum(1 for level in self.risk_levels if level == "MEDIUM")
        low_count = sum(1 for level in self.risk_levels if level == "LOW")
        return {
            "device": self.device,
            "frames_processed": len(self.risk_levels),
            "aligned_pairs": len(self.sync_gaps_ms),
            "average_sync_gap_ms": round(mean(self.sync_gaps_ms), 5) if self.sync_gaps_ms else 0.0,
            "average_inference_latency_ms": round(mean(self.inference_latencies_ms), 5) if self.inference_latencies_ms else 0.0,
            "p95_inference_latency_ms": self._p95(self.inference_latencies_ms),
            "high_risk_events": high_count,
            "medium_risk_events": med_count,
            "low_risk_events": low_count,
        }


def write_timeline(events: list[FusedEvent], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    timeline_path = output_dir / "timeline.jsonl"
    with timeline_path.open("w") as f:
        for event in events:
            f.write(json.dumps(asdict(event)) + "\n")


def write_summary(summary: dict[str, Any], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
