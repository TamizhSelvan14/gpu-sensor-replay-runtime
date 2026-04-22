# GPU-Accelerated Sensor Replay & Inference Runtime (C++)

This project is a **C++ runtime** for replaying timestamped multimodal sensor data, synchronizing streams, running an inference stage, fusing outputs with motion context, and writing validation outputs.

## What it does
- Replays recorded **camera** and **IMU** streams with original timestamp spacing.
- Synchronizes camera frames with nearest IMU samples within a configurable tolerance.
- Runs an inference backend over aligned samples.
- Fuses model output with motion context into a risk event.
- Writes `timeline.jsonl` and `summary.json` for offline validation.

## Why it exists
Real AI systems are more than models. They need replay, synchronization, timing, observability, and reproducibility. This project focuses on the runtime side of AI systems.

## Current state
- Core runtime: **C++17**
- Replay + synchronization: **C++ threads + queues**
- Inference backend: **CPU backend in C++**
- CUDA support: **optional hook via `-DUSE_CUDA=ON` placeholder**, ready to extend

## Build
```bash
cmake -S . -B build
cmake --build build -j
```

## Generate sample data
```bash
./build/gpu_sensor_runtime generate-sample-data --output-dir sample_data
```

## Run the runtime
```bash
./build/gpu_sensor_runtime run --data-dir sample_data --output-dir outputs --device auto --replay-speed 20 --sync-tolerance-ms 40
```

## Outputs
- `outputs/timeline.jsonl` — one fused event per line
- `outputs/summary.json` — aggregate metrics

## High-level architecture
Recorded Data -> Replay Threads -> Sync Coordinator -> Inference Backend -> Fusion Layer -> Metrics/Outputs

## Repo structure
- `include/gpu_runtime/` — headers for types, queues, replay, sync, inference, fusion
- `src/` — runtime implementation
- `sample_data/` — generated sensor data after running generator
- `outputs/` — written at runtime

## Notes
The current inference backend is intentionally lightweight and deterministic. The project is structured so that a real ONNX/TensorRT/CUDA backend can be added later without changing the rest of the runtime design.
