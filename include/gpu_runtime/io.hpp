#pragma once

#include <string>
#include <vector>

#include "gpu_runtime/types.hpp"

namespace gpu_runtime {

void generate_sample_data(const std::string& output_dir, int duration_seconds = 8,
                          int camera_fps = 10, int imu_hz = 50);

std::vector<CameraPacket> load_camera_packets(const std::string& data_dir);
std::vector<IMUPacket> load_imu_packets(const std::string& data_dir);

Summary write_outputs(const std::vector<FusedEvent>& events, const std::string& output_dir);

Summary run_pipeline(const RunOptions& options);

}  // namespace gpu_runtime
