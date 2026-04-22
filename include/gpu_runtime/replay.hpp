#pragma once

#include <chrono>
#include <thread>
#include <vector>

#include "gpu_runtime/threadsafe_queue.hpp"
#include "gpu_runtime/types.hpp"

namespace gpu_runtime {

template <typename T>
struct ReplayMessage {
  bool end_of_stream{false};
  T payload{};
};

template <typename T>
class ReplayThread {
 public:
  ReplayThread(const std::vector<T>& packets, double replay_speed,
               ThreadSafeQueue<ReplayMessage<T>>& queue)
      : packets_(packets), replay_speed_(replay_speed), queue_(queue) {}

  void start() {
    worker_ = std::thread([this] { run(); });
  }

  void join() {
    if (worker_.joinable()) worker_.join();
  }

 private:
  void run() {
    if (packets_.empty()) {
      queue_.push(ReplayMessage<T>{true, T{}});
      return;
    }

    queue_.push(ReplayMessage<T>{false, packets_[0]});
    for (std::size_t i = 1; i < packets_.size(); ++i) {
      auto delta = static_cast<double>(packets_[i].timestamp_ms - packets_[i - 1].timestamp_ms);
      auto sleep_ms = delta / replay_speed_;
      if (sleep_ms > 0.0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(static_cast<int>(sleep_ms)));
      }
      queue_.push(ReplayMessage<T>{false, packets_[i]});
    }
    queue_.push(ReplayMessage<T>{true, T{}});
  }

  const std::vector<T>& packets_;
  double replay_speed_;
  ThreadSafeQueue<ReplayMessage<T>>& queue_;
  std::thread worker_;
};

}  // namespace gpu_runtime
