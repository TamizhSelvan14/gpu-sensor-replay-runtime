# GPU-Accelerated Sensor Replay & Inference Runtime

A production-style runtime for deterministic replay, synchronization, GPU-backed inference, and fusion over timestamped sensor streams.

## Project abstract
Modern autonomous systems, robotics stacks, and edge AI applications need more than model accuracy: they need a runtime that can **replay real sensor traffic deterministically**, **align asynchronous streams**, **run inference with predictable latency**, and **debug failures offline**. This project models that control-plane/runtime problem.

The demo version uses:
- a camera stream (`PNG` frames + timestamps)
- an IMU stream (`CSV` timestamps + speed/acceleration)
- a multithreaded replay engine
- a synchronization coordinator with tolerance windows
- a GPU-aware inference backend (PyTorch with CUDA auto-detection)
- a fusion layer that combines visual and motion context
- metrics, timeline traces, and replay diagnostics

The included inference backend is intentionally lightweight and deterministic so the project runs anywhere. The runtime is designed so you can swap in a real ONNX / TensorRT / PyTorch model without changing the replay or synchronization layers.

## Why this is useful
This runtime is useful for:
- offline validation of AI pipelines against recorded sensor data
- debugging drift, skew, queueing delays, and inference latency
- testing fusion logic with reproducible replay
- evaluating CPU vs GPU execution paths under the same input stream
- simulating production issues before deploying to a live edge / vehicle system

## High-level architecture

```text
Sensor Files / Streams
   -> Replay Loader Threads
   -> Sync Coordinator
   -> Preprocessing Stage
   -> GPU / CPU Inference Worker
   -> Fusion Layer
   -> Metrics + Debug Timeline
```

More detail:

```text
+----------------------+     +----------------------+ 
| Camera Frames        |     | IMU Samples          |
| timestamps + PNGs    |     | timestamps + CSV     |
+----------+-----------+     +----------+-----------+
           |                              |
           v                              v
+----------------------+     +----------------------+
| Camera Replay Thread |     | IMU Replay Thread    |
+----------+-----------+     +----------+-----------+
           \                             /
            \                           /
             v                         v
              +----------------------+
              | Sync Coordinator     |
              | nearest-neighbor     |
              | alignment window     |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Inference Backend    |
              | Torch / CPU / CUDA   |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Fusion Layer         |
              | label + motion risk  |
              +----------+-----------+
                         |
                         v
              +----------------------+
              | Metrics + Timeline   |
              | latency / drift /    |
              | throughput / events  |
              +----------------------+
```

## Features
- Deterministic timestamp-based replay
- Multithreaded sensor loaders
- Nearest-neighbor synchronization with configurable tolerance
- GPU-aware inference backend (`cuda` when available, CPU fallback)
- Fused event scoring using image + IMU context
- Timeline JSONL output for debugging
- Summary metrics including p95 inference latency and average sync gap
- Synthetic sample data generator for immediate local runs

## Tech stack
- Python 3.10+
- NumPy
- Pillow
- PyTorch (optional GPU via CUDA)

## Quickstart

### 1) Create a virtual environment
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2) Generate sample data
```bash
python main.py generate-sample-data --output-dir sample_data
```

### 3) Run the replay + inference pipeline
```bash
python main.py run --data-dir sample_data --device auto --replay-speed 20
```

### 4) Review outputs
- `outputs/timeline.jsonl`
- `outputs/summary.json`

## Example output
```text
Device: cuda
Frames processed: 120
Aligned pairs: 120
Average sync gap (ms): 7.10
Average inference latency (ms): 0.89
P95 inference latency (ms): 1.41
High risk events: 17
```

## Repository layout
```text
gpu_sensor_runtime/
  docs/
  sample_data/
  src/gpu_sensor_runtime/
    data/
    replay/
    inference/
    fusion/
    observability/
  tests/
  main.py
  requirements.txt
```

