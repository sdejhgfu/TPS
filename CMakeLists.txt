cmake_minimum_required(VERSION 3.16)

project(TPSInference LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimize for 2-socket, 8-core (16 total), 32GB DDR5 system
add_compile_options(-O3 -march=x86-64 -mavx2 -mfma -fno-exceptions -fno-rtti -flto)
add_compile_options(-funroll-loops -ffast-math -fno-common -Wall -Wextra -Wno-unused-parameter)
add_compile_options(-fopenmp)  # Enable OpenMP for multi-core utilization

# Set thread count for 16 cores (2 sockets * 8 cores)
set(CMAKE_THREAD_PREFER_PTHREAD ON)
find_package(Threads REQUIRED)

# Create the core library
add_library(tps_core
    src/inference_engine.cpp
)
target_include_directories(tps_core PUBLIC src)
target_link_libraries(tps_core PRIVATE Threads::Threads)

# Main benchmark executable
add_executable(tps
    src/main.cpp
)
target_link_libraries(tps PRIVATE tps_core)

# Test executable
add_executable(tps_tests
    src/test.cpp
)
target_link_libraries(tps_tests PRIVATE tps_core)

# Install targets
install(TARGETS tps tps_tests RUNTIME DESTINATION bin)