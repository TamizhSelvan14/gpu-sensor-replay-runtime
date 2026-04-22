#include "gpu_runtime/io.hpp"

#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>

namespace gpu_runtime {
namespace {

std::unordered_map<std::string, std::string> parse_args(int argc, char** argv, int start_index) {
  std::unordered_map<std::string, std::string> args;
  for (int i = start_index; i < argc; ++i) {
    std::string token = argv[i];
    if (token.rfind("--", 0) == 0 && i + 1 < argc) {
      args[token] = argv[++i];
    }
  }
  return args;
}

void print_usage() {
  std::cout << "Usage:\n"
            << "  gpu_sensor_runtime generate-sample-data --output-dir sample_data\n"
            << "  gpu_sensor_runtime run --data-dir sample_data --output-dir outputs --device auto --replay-speed 20 --sync-tolerance-ms 40\n";
}

}  // namespace
}  // namespace gpu_runtime

int main(int argc, char** argv) {
  using namespace gpu_runtime;

  if (argc < 2) {
    print_usage();
    return 1;
  }

  const std::string command = argv[1];
  auto args = parse_args(argc, argv, 2);

  try {
    if (command == "generate-sample-data") {
      const auto output_dir = args.count("--output-dir") ? args["--output-dir"] : "sample_data";
      generate_sample_data(output_dir);
      return 0;
    }

    if (command == "run") {
      RunOptions options;
      options.data_dir = args.count("--data-dir") ? args["--data-dir"] : "sample_data";
      options.output_dir = args.count("--output-dir") ? args["--output-dir"] : "outputs";
      options.device = args.count("--device") ? args["--device"] : "auto";
      if (args.count("--replay-speed")) options.replay_speed = std::stod(args["--replay-speed"]);
      if (args.count("--sync-tolerance-ms")) options.sync_tolerance_ms = std::stod(args["--sync-tolerance-ms"]);
      run_pipeline(options);
      return 0;
    }

    print_usage();
    return 1;
  } catch (const std::exception& ex) {
    std::cerr << "Error: " << ex.what() << "\n";
    return 1;
  }
}
