#include "gpu_runtime/fusion.hpp"
#include "gpu_runtime/inference.hpp"
#include "gpu_runtime/io.hpp"
#include "gpu_runtime/replay.hpp"
#include "gpu_runtime/sync.hpp"
#include "gpu_runtime/threadsafe_queue.hpp"
#include "gpu_runtime/types.hpp"

#include <deque>
#include <iostream>
#include <optional>
#include <vector>

namespace gpu_runtime {

Summary run_pipeline(const RunOptions& options) {
  const auto camera_packets = load_camera_packets(options.data_dir);
  const auto imu_packets = load_imu_packets(options.data_dir);

  ThreadSafeQueue<ReplayMessage<CameraPacket>> camera_queue;
  ThreadSafeQueue<ReplayMessage<IMUPacket>> imu_queue;

  ReplayThread<CameraPacket> camera_thread(camera_packets, options.replay_speed, camera_queue);
  ReplayThread<IMUPacket> imu_thread(imu_packets, options.replay_speed, imu_queue);

  camera_thread.start();
  imu_thread.start();

  SyncCoordinator coordinator(options.sync_tolerance_ms);
  auto backend = create_inference_backend(options.device);
  EventFuser fuser;
  std::vector<FusedEvent> events;
  std::deque<CameraPacket> pending_cameras;

  bool camera_done = false;
  bool imu_done = false;

  auto process_pending = [&]() {
    std::deque<CameraPacket> still_pending;
    while (!pending_cameras.empty()) {
      CameraPacket cam = pending_cameras.front();
      pending_cameras.pop_front();
      auto synced = coordinator.try_sync_camera(cam);
      if (!synced.has_value()) {
        still_pending.push_back(cam);
        continue;
      }
      auto inference = backend->infer(*synced);
      auto event = fuser.fuse(*synced, inference);
      events.push_back(event);
    }
    pending_cameras = std::move(still_pending);
  };

  while (!(camera_done && imu_done)) {
    if (!imu_done) {
      auto msg = imu_queue.pop();
      if (msg.end_of_stream) {
        imu_done = true;
      } else {
        coordinator.ingest_imu(msg.payload);
        process_pending();
      }
    }

    if (!camera_done) {
      auto msg = camera_queue.pop();
      if (msg.end_of_stream) {
        camera_done = true;
      } else {
        pending_cameras.push_back(msg.payload);
        process_pending();
      }
    }
  }

  process_pending();

  camera_thread.join();
  imu_thread.join();

  Summary summary = write_outputs(events, options.output_dir);
  std::cout << "--- Replay Summary ---\n";
  std::cout << "device: " << summary.device << "\n";
  std::cout << "frames_processed: " << summary.frames_processed << "\n";
  std::cout << "aligned_pairs: " << summary.aligned_pairs << "\n";
  std::cout << "average_sync_gap_ms: " << summary.average_sync_gap_ms << "\n";
  std::cout << "average_inference_latency_ms: " << summary.average_inference_latency_ms << "\n";
  std::cout << "p95_inference_latency_ms: " << summary.p95_inference_latency_ms << "\n";
  std::cout << "high_risk_events: " << summary.high_risk_events << "\n";
  std::cout << "medium_risk_events: " << summary.medium_risk_events << "\n";
  std::cout << "low_risk_events: " << summary.low_risk_events << "\n";

  return summary;
}

}  // namespace gpu_runtime
