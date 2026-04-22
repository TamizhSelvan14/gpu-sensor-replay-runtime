#include "gpu_runtime/fusion.hpp"

#include <algorithm>
#include <cmath>

namespace gpu_runtime {

FusedEvent EventFuser::fuse(const SyncedSample& sample, const InferenceResult& inference) const {
  double score = 0.0;
  if (inference.label == Label::Vehicle) {
    score = 0.50 * inference.confidence + 0.02 * sample.imu.speed + 0.20 * std::abs(sample.imu.acceleration);
  } else if (inference.label == Label::Pedestrian) {
    score = 0.55 * inference.confidence + 0.015 * sample.imu.speed + 0.30 * std::max(0.0, sample.imu.acceleration);
  } else {
    score = 0.20 * inference.confidence + 0.005 * sample.imu.speed;
  }
  score = std::clamp(score, 0.0, 1.0);

  std::string level = "LOW";
  if (score >= 0.72) level = "HIGH";
  else if (score >= 0.45) level = "MEDIUM";

  FusedEvent event;
  event.timestamp_ms = sample.camera.timestamp_ms;
  event.label = inference.label;
  event.confidence = inference.confidence;
  event.speed = sample.imu.speed;
  event.acceleration = sample.imu.acceleration;
  event.yaw_rate = sample.imu.yaw_rate;
  event.sync_gap_ms = sample.sync_gap_ms;
  event.inference_latency_ms = inference.latency_ms;
  event.risk_score = score;
  event.risk_level = level;
  event.device = inference.device;
  event.mean_red = inference.mean_red;
  event.mean_green = inference.mean_green;
  event.mean_blue = inference.mean_blue;
  event.brightness = inference.brightness;
  return event;
}

}  // namespace gpu_runtime
