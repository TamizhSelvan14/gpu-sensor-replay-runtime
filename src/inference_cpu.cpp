#include "gpu_runtime/inference.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <vector>

namespace gpu_runtime {

namespace {

struct ImageFeatures {
  double mean_red{};
  double mean_green{};
  double mean_blue{};
  double brightness{};
};

ImageFeatures read_ppm_features(const std::string& path) {
  std::ifstream in(path);
  if (!in) throw std::runtime_error("Failed to open image: " + path);

  std::string magic;
  int width = 0, height = 0, maxv = 0;
  in >> magic >> width >> height >> maxv;
  if (magic != "P3") throw std::runtime_error("Unsupported PPM format");

  long long count = 0;
  double rsum = 0.0, gsum = 0.0, bsum = 0.0;
  int r, g, b;
  while (in >> r >> g >> b) {
    rsum += r;
    gsum += g;
    bsum += b;
    ++count;
  }
  if (count == 0) return {};

  ImageFeatures f;
  f.mean_red = rsum / count / 255.0;
  f.mean_green = gsum / count / 255.0;
  f.mean_blue = bsum / count / 255.0;
  f.brightness = (f.mean_red + f.mean_green + f.mean_blue) / 3.0;
  return f;
}

std::vector<double> softmax(const std::vector<double>& logits) {
  double max_logit = *std::max_element(logits.begin(), logits.end());
  std::vector<double> exps;
  double sum = 0.0;
  for (double x : logits) {
    double e = std::exp(x - max_logit);
    exps.push_back(e);
    sum += e;
  }
  for (double& e : exps) e /= sum;
  return exps;
}

class CpuInferenceBackend final : public InferenceBackend {
 public:
  InferenceResult infer(const SyncedSample& sample) override {
    auto start = std::chrono::high_resolution_clock::now();
    ImageFeatures f = read_ppm_features(sample.camera.image_path);

    double vehicle_logit = 2.3 * f.mean_red + 0.08 * sample.imu.speed + 0.25 * std::abs(sample.imu.acceleration);
    double pedestrian_logit = 2.3 * f.mean_green + 0.15 * std::max(0.0, sample.imu.acceleration) + 0.1;
    double clear_logit = 2.0 * f.mean_blue + 1.0 * f.brightness - 0.03 * sample.imu.speed;

    std::vector<double> probs = softmax({clear_logit, vehicle_logit, pedestrian_logit});
    auto best = std::max_element(probs.begin(), probs.end());
    int idx = static_cast<int>(std::distance(probs.begin(), best));

    auto end = std::chrono::high_resolution_clock::now();
    double latency = std::chrono::duration<double, std::milli>(end - start).count();

    InferenceResult result;
    result.label = idx == 0 ? Label::Clear : (idx == 1 ? Label::Vehicle : Label::Pedestrian);
    result.confidence = *best;
    result.device = "cpu";
    result.latency_ms = latency;
    result.mean_red = f.mean_red;
    result.mean_green = f.mean_green;
    result.mean_blue = f.mean_blue;
    result.brightness = f.brightness;
    return result;
  }

  std::string device() const override { return "cpu"; }
};

}  // namespace

std::unique_ptr<InferenceBackend> create_inference_backend(const std::string& requested_device) {
  (void)requested_device;
  return std::make_unique<CpuInferenceBackend>();
}

}  // namespace gpu_runtime
