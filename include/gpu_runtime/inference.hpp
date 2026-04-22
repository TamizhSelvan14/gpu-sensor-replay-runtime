#pragma once

#include <memory>
#include <string>

#include "gpu_runtime/types.hpp"

namespace gpu_runtime {

class InferenceBackend {
 public:
  virtual ~InferenceBackend() = default;
  virtual InferenceResult infer(const SyncedSample& sample) = 0;
  virtual std::string device() const = 0;
};

std::unique_ptr<InferenceBackend> create_inference_backend(const std::string& requested_device);

}  // namespace gpu_runtime
