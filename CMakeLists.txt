cmake_minimum_required(VERSION 3.16)
project(gpu_sensor_runtime LANGUAGES CXX)

option(USE_CUDA "Enable optional CUDA backend" OFF)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include_directories(include)

set(SOURCES
    src/main.cpp
    src/data_generator.cpp
    src/loader.cpp
    src/runtime.cpp
    src/inference_cpu.cpp
    src/fusion.cpp
)

if(USE_CUDA)
  enable_language(CUDA)
  add_compile_definitions(GPU_RUNTIME_USE_CUDA=1)
  list(APPEND SOURCES src/inference_cuda.cu)
else()
  add_compile_definitions(GPU_RUNTIME_USE_CUDA=0)
endif()

add_executable(gpu_sensor_runtime ${SOURCES})

if(USE_CUDA)
  set_target_properties(gpu_sensor_runtime PROPERTIES CUDA_STANDARD 14 CUDA_STANDARD_REQUIRED ON)
endif()
