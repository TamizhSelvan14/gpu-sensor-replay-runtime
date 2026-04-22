#pragma once

#include <condition_variable>
#include <mutex>
#include <queue>

namespace gpu_runtime {

template <typename T>
class ThreadSafeQueue {
 public:
  void push(const T& item) {
    {
      std::lock_guard<std::mutex> lock(mutex_);
      queue_.push(item);
    }
    cv_.notify_one();
  }

  T pop() {
    std::unique_lock<std::mutex> lock(mutex_);
    cv_.wait(lock, [this] { return !queue_.empty(); });
    T value = queue_.front();
    queue_.pop();
    return value;
  }

 private:
  std::queue<T> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
};

}  // namespace gpu_runtime
