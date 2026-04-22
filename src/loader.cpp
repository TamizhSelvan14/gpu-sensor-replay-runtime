#include "gpu_runtime/io.hpp"

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>

namespace fs = std::filesystem;

namespace gpu_runtime {

namespace {

std::vector<std::string> split_csv(const std::string& line) {
  std::stringstream ss(line);
  std::string item;
  std::vector<std::string> parts;
  while (std::getline(ss, item, ',')) {
    parts.push_back(item);
  }
  return parts;
}

}  // namespace

std::vector<CameraPacket> load_camera_packets(const std::string& data_dir) {
  std::ifstream in(fs::path(data_dir) / "camera_stream.csv");
  if (!in) throw std::runtime_error("Failed to open camera_stream.csv");

  std::vector<CameraPacket> packets;
  std::string line;
  std::getline(in, line);  // header
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto parts = split_csv(line);
    CameraPacket packet;
    packet.sequence_id = std::stoi(parts.at(0));
    packet.timestamp_ms = std::stoll(parts.at(1));
    packet.image_path = (fs::path(data_dir) / parts.at(2)).string();
    packets.push_back(packet);
  }
  return packets;
}

std::vector<IMUPacket> load_imu_packets(const std::string& data_dir) {
  std::ifstream in(fs::path(data_dir) / "imu_stream.csv");
  if (!in) throw std::runtime_error("Failed to open imu_stream.csv");

  std::vector<IMUPacket> packets;
  std::string line;
  std::getline(in, line);  // header
  while (std::getline(in, line)) {
    if (line.empty()) continue;
    auto parts = split_csv(line);
    IMUPacket packet;
    packet.sequence_id = std::stoi(parts.at(0));
    packet.timestamp_ms = std::stoll(parts.at(1));
    packet.speed = std::stod(parts.at(2));
    packet.acceleration = std::stod(parts.at(3));
    packet.yaw_rate = std::stod(parts.at(4));
    packets.push_back(packet);
  }
  return packets;
}

Summary write_outputs(const std::vector<FusedEvent>& events, const std::string& output_dir) {
  fs::create_directories(output_dir);

  std::ofstream timeline(fs::path(output_dir) / "timeline.jsonl");
  std::vector<double> latencies;
  double gap_sum = 0.0;
  double latency_sum = 0.0;
  std::size_t high = 0, med = 0, low = 0;
  std::string device = "cpu";

  for (const auto& event : events) {
    timeline << "{"
             << "\"timestamp_ms\":" << event.timestamp_ms << ','
             << "\"label\":\"" << label_to_string(event.label) << "\"," 
             << "\"confidence\":" << event.confidence << ','
             << "\"speed\":" << event.speed << ','
             << "\"acceleration\":" << event.acceleration << ','
             << "\"yaw_rate\":" << event.yaw_rate << ','
             << "\"sync_gap_ms\":" << event.sync_gap_ms << ','
             << "\"inference_latency_ms\":" << event.inference_latency_ms << ','
             << "\"risk_score\":" << event.risk_score << ','
             << "\"risk_level\":\"" << event.risk_level << "\"," 
             << "\"device\":\"" << event.device << "\""
             << "}\n";

    gap_sum += event.sync_gap_ms;
    latency_sum += event.inference_latency_ms;
    latencies.push_back(event.inference_latency_ms);
    device = event.device;
    if (event.risk_level == "HIGH") ++high;
    else if (event.risk_level == "MEDIUM") ++med;
    else ++low;
  }

  std::sort(latencies.begin(), latencies.end());
  double p95 = latencies.empty() ? 0.0 : latencies[static_cast<std::size_t>(0.95 * (latencies.size() - 1))];

  Summary summary;
  summary.device = device;
  summary.frames_processed = events.size();
  summary.aligned_pairs = events.size();
  summary.average_sync_gap_ms = events.empty() ? 0.0 : gap_sum / events.size();
  summary.average_inference_latency_ms = events.empty() ? 0.0 : latency_sum / events.size();
  summary.p95_inference_latency_ms = p95;
  summary.high_risk_events = high;
  summary.medium_risk_events = med;
  summary.low_risk_events = low;

  std::ofstream out(fs::path(output_dir) / "summary.json");
  out << "{\n"
      << "  \"device\": \"" << summary.device << "\",\n"
      << "  \"frames_processed\": " << summary.frames_processed << ",\n"
      << "  \"aligned_pairs\": " << summary.aligned_pairs << ",\n"
      << "  \"average_sync_gap_ms\": " << summary.average_sync_gap_ms << ",\n"
      << "  \"average_inference_latency_ms\": " << summary.average_inference_latency_ms << ",\n"
      << "  \"p95_inference_latency_ms\": " << summary.p95_inference_latency_ms << ",\n"
      << "  \"high_risk_events\": " << summary.high_risk_events << ",\n"
      << "  \"medium_risk_events\": " << summary.medium_risk_events << ",\n"
      << "  \"low_risk_events\": " << summary.low_risk_events << "\n"
      << "}\n";

  return summary;
}

}  // namespace gpu_runtime
