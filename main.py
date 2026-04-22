from __future__ import annotations

import argparse
from pathlib import Path

from src.gpu_sensor_runtime.data.generate_sample_data import generate_sample_data
from src.gpu_sensor_runtime.runtime import run_pipeline


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="GPU-Accelerated Sensor Replay & Inference Runtime")
    subparsers = parser.add_subparsers(dest="command", required=True)

    gen = subparsers.add_parser("generate-sample-data", help="Generate synthetic sensor streams")
    gen.add_argument("--output-dir", type=Path, default=Path("sample_data"))
    gen.add_argument("--frames", type=int, default=120)
    gen.add_argument("--frame-rate", type=float, default=10.0)
    gen.add_argument("--imu-rate", type=float, default=50.0)

    run = subparsers.add_parser("run", help="Run replay + inference + fusion")
    run.add_argument("--data-dir", type=Path, default=Path("sample_data"))
    run.add_argument("--output-dir", type=Path, default=Path("outputs"))
    run.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    run.add_argument("--sync-tolerance-ms", type=float, default=30.0)
    run.add_argument("--replay-speed", type=float, default=20.0)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "generate-sample-data":
        generate_sample_data(
            output_dir=args.output_dir,
            frame_count=args.frames,
            frame_rate_hz=args.frame_rate,
            imu_rate_hz=args.imu_rate,
        )
        print(f"Sample data generated at {args.output_dir}")
    elif args.command == "run":
        summary = run_pipeline(
            data_dir=args.data_dir,
            output_dir=args.output_dir,
            device=args.device,
            sync_tolerance_ms=args.sync_tolerance_ms,
            replay_speed=args.replay_speed,
        )
        print("--- Replay Summary ---")
        for key, value in summary.items():
            print(f"{key}: {value}")


if __name__ == "__main__":
    main()
