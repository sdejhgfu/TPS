cmake_minimum_required(VERSION 3.16)
project(HighPerformanceInferenceEngine VERSION 1.0.0 LANGUAGES CXX C)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Compiler-specific options
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -flto -fno-exceptions")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic")
endif()

# Include directories
include_directories(
    src
    src/engine
    src/memory
    src/quantization
    src/cache
    src/utils
)

# Source files
set(ENGINE_SOURCES
    src/engine/inference_engine.cpp
    src/engine/tokenizer.cpp
    src/engine/model_loader.cpp
    src/engine/sampler.cpp
    src/memory/memory_manager.cpp
    src/memory/page_manager.cpp
    src/memory/memory_pool.cpp
    src/quantization/quantizer.cpp
    src/quantization/int8_quantizer.cpp
    src/quantization/int4_quantizer.cpp
    src/cache/kv_cache.cpp
    src/cache/attention_cache.cpp
    src/utils/math_utils.cpp
    src/utils/timer.cpp
    src/utils/logger.cpp
)

# Main executable
add_executable(inference_engine
    src/main.cpp
    ${ENGINE_SOURCES}
)

# Test executable
file(GLOB TEST_SOURCES "tests/*.cpp")
add_executable(test_runner
    ${TEST_SOURCES}
    ${ENGINE_SOURCES}
)

# Benchmark executable
add_executable(benchmark
    src/benchmark.cpp
    ${ENGINE_SOURCES}
)

# Link libraries
target_link_libraries(inference_engine)
target_link_libraries(test_runner)
target_link_libraries(benchmark)

# Compiler definitions
target_compile_definitions(inference_engine PRIVATE
    OPTIMIZE_FOR_CPU
    ENABLE_SIMD
    ENABLE_MULTITHREADING
)

# Enable testing
enable_testing()
add_test(NAME unit_tests COMMAND test_runner)

# Installation
install(TARGETS inference_engine test_runner benchmark
    RUNTIME DESTINATION bin
)

# CPack configuration
set(CPACK_PACKAGE_NAME "HighPerformanceInferenceEngine")
set(CPACK_PACKAGE_VERSION "1.0.0")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-Performance Inference Engine for GPT-OSS-20B")
include(CPack)

