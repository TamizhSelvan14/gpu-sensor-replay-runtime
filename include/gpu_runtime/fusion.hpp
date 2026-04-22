#pragma once

#include "gpu_runtime/types.hpp"

namespace gpu_runtime {

class EventFuser {
 public:
  FusedEvent fuse(const SyncedSample& sample, const InferenceResult& inference) const;
};

}  // namespace gpu_runtime
