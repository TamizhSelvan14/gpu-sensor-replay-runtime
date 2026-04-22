#pragma once

#include <cmath>
#include <deque>
#include <optional>

#include "gpu_runtime/types.hpp"

namespace gpu_runtime {

class SyncCoordinator {
 public:
  explicit SyncCoordinator(double tolerance_ms) : tolerance_ms_(tolerance_ms) {}

  void ingest_imu(const IMUPacket& packet) {
    imu_buffer_.push_back(packet);
    while (imu_buffer_.size() > max_buffer_size_) {
      imu_buffer_.pop_front();
    }
  }

  std::optional<SyncedSample> try_sync_camera(const CameraPacket& packet) const {
    if (imu_buffer_.empty()) return std::nullopt;

    const IMUPacket* best = nullptr;
    double best_gap = 1e18;
    for (const auto& imu : imu_buffer_) {
      double gap = std::abs(static_cast<double>(packet.timestamp_ms - imu.timestamp_ms));
      if (gap < best_gap) {
        best_gap = gap;
        best = &imu;
      }
    }
    if (!best || best_gap > tolerance_ms_) return std::nullopt;
    return SyncedSample{packet, *best, best_gap};
  }

 private:
  double tolerance_ms_;
  std::size_t max_buffer_size_{256};
  std::deque<IMUPacket> imu_buffer_;
};

}  // namespace gpu_runtime
