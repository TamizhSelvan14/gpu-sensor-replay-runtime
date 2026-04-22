#pragma once

#include <cstdint>
#include <deque>
#include <optional>
#include <string>
#include <vector>

namespace gpu_runtime {

enum class Label {
  Clear,
  Vehicle,
  Pedestrian
};

inline std::string label_to_string(Label label) {
  switch (label) {
    case Label::Clear: return "clear";
    case Label::Vehicle: return "vehicle";
    case Label::Pedestrian: return "pedestrian";
  }
  return "unknown";
}

struct CameraPacket {
  std::int64_t timestamp_ms{};
  int sequence_id{};
  std::string image_path;
};

struct IMUPacket {
  std::int64_t timestamp_ms{};
  int sequence_id{};
  double speed{};
  double acceleration{};
  double yaw_rate{};
};

struct SyncedSample {
  CameraPacket camera;
  IMUPacket imu;
  double sync_gap_ms{};
};

struct InferenceResult {
  Label label{Label::Clear};
  double confidence{};
  std::string device{"cpu"};
  double latency_ms{};
  double mean_red{};
  double mean_green{};
  double mean_blue{};
  double brightness{};
};

struct FusedEvent {
  std::int64_t timestamp_ms{};
  Label label{Label::Clear};
  double confidence{};
  double speed{};
  double acceleration{};
  double yaw_rate{};
  double sync_gap_ms{};
  double inference_latency_ms{};
  double risk_score{};
  std::string risk_level;
  std::string device;
  double mean_red{};
  double mean_green{};
  double mean_blue{};
  double brightness{};
};

struct Summary {
  std::string device{"cpu"};
  std::size_t frames_processed{};
  std::size_t aligned_pairs{};
  double average_sync_gap_ms{};
  double average_inference_latency_ms{};
  double p95_inference_latency_ms{};
  std::size_t high_risk_events{};
  std::size_t medium_risk_events{};
  std::size_t low_risk_events{};
};

struct RunOptions {
  std::string data_dir;
  std::string output_dir;
  std::string device{"auto"};
  double replay_speed{20.0};
  double sync_tolerance_ms{40.0};
};

}  // namespace gpu_runtime
