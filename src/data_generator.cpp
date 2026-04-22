#include "gpu_runtime/io.hpp"

#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <random>
#include <sstream>

namespace fs = std::filesystem;

namespace gpu_runtime {

namespace {

Label label_for_index(int i, int total) {
  if (i < total / 3) return Label::Clear;
  if (i < (2 * total) / 3) return Label::Vehicle;
  return Label::Pedestrian;
}

std::tuple<int, int, int> rgb_for_label(Label label) {
  switch (label) {
    case Label::Clear: return {80, 120, 210};
    case Label::Vehicle: return {210, 70, 70};
    case Label::Pedestrian: return {80, 210, 110};
  }
  return {128, 128, 128};
}

void write_ppm(const fs::path& path, Label label, int frame_index) {
  constexpr int width = 64;
  constexpr int height = 64;
  auto [r_base, g_base, b_base] = rgb_for_label(label);

  std::ofstream out(path);
  out << "P3\n" << width << " " << height << "\n255\n";
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      int r = 20;
      int g = 20;
      int b = 20;

      if (y > 44) {
        r = 50;
        g = 50;
        b = 50;
      }
      if (x == 20 || x == 44) {
        r = 200;
        g = 200;
        b = 200;
      }

      int cx = 18 + (frame_index % 18);
      int cy = 28;
      if (x > cx && x < cx + 18 && y > cy && y < cy + 18) {
        r = r_base;
        g = g_base;
        b = b_base;
      }

      out << r << ' ' << g << ' ' << b << ' ';
    }
    out << '\n';
  }
}

}  // namespace

void generate_sample_data(const std::string& output_dir, int duration_seconds, int camera_fps, int imu_hz) {
  fs::create_directories(fs::path(output_dir) / "camera_frames");

  const int camera_count = duration_seconds * camera_fps;
  const int imu_count = duration_seconds * imu_hz;
  const int camera_interval_ms = 1000 / camera_fps;
  const int imu_interval_ms = 1000 / imu_hz;

  std::ofstream camera_csv(fs::path(output_dir) / "camera_stream.csv");
  std::ofstream imu_csv(fs::path(output_dir) / "imu_stream.csv");

  camera_csv << "sequence_id,timestamp_ms,image_path\n";
  imu_csv << "sequence_id,timestamp_ms,speed,acceleration,yaw_rate\n";

  for (int i = 0; i < camera_count; ++i) {
    Label label = label_for_index(i, camera_count);
    std::ostringstream name;
    name << "frame_" << std::setw(4) << std::setfill('0') << i << ".ppm";
    fs::path rel = fs::path("camera_frames") / name.str();
    write_ppm(fs::path(output_dir) / rel, label, i);
    camera_csv << i << ',' << i * camera_interval_ms << ',' << rel.string() << "\n";
  }

  for (int i = 0; i < imu_count; ++i) {
    double t = static_cast<double>(i) / imu_hz;
    double speed = 12.0 + 5.0 * std::sin(t * 1.2);
    double accel = 0.8 * std::cos(t * 1.8);
    double yaw = 0.2 * std::sin(t * 2.5);
    imu_csv << i << ',' << i * imu_interval_ms << ',' << speed << ',' << accel << ',' << yaw << "\n";
  }

  std::cout << "Generated sample data in: " << output_dir << "\n";
}

}  // namespace gpu_runtime
