# Architecture notes

## Components

### 1. Replay loader threads
Each stream is replayed on its own thread. Packets preserve source timestamps and are released according to a replay-speed multiplier. This gives the runtime realistic inter-arrival patterns without waiting full wall-clock duration.

### 2. Synchronization coordinator
The sync layer stores a rolling IMU buffer and aligns each camera frame to the nearest IMU sample within a tolerance window. In production this could be extended to interpolation, watermark-based synchronization, or multi-stream barrier logic.

### 3. GPU-aware inference backend
The demo backend uses a deterministic tensor-based classifier implemented in PyTorch. If CUDA is available, tensors are moved to GPU automatically. The backend contract is intentionally simple so a real ONNX Runtime / TensorRT / TorchScript model can replace it.

### 4. Fusion layer
Fusion combines image-level predictions with motion context (speed, acceleration) to produce a risk score and risk level. The point is not model sophistication; it is demonstrating the runtime path from aligned sensor sample to actionable output.

### 5. Metrics and timeline
Every fused event records sync gap, inference latency, label, and risk score. This provides replay diagnostics and helps reason about drift, queueing, and runtime behavior under different replay speeds.

## How to extend this project
- Replace the demo classifier with a real ONNX object detector
- Add a second camera or lidar-like stream
- Add batch inference
- Add Prometheus export
- Add gRPC workers for remote GPU inference
- Add temporal watermarking and dropped-frame diagnostics
