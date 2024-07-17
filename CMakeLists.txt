cmake_minimum_required(VERSION 3.16)
project(fastmultithreshold)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -march=native -mtune=native -Wall")
set(CMAKE_C_FLAGS_RELEASE   "${CMAKE_C_FLAGS_RELEASE}   -O3")

set(default_build_type "Release")
if (NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to '${default_build_type}' as none was specified.")
    set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build." FORCE)
endif ()
message(STATUS "build type: ${CMAKE_BUILD_TYPE}")

include(FetchContent)
FetchContent_Declare(
  googlebenchmark
  GIT_REPOSITORY  https://github.com/google/benchmark.git
  GIT_TAG         main
  GIT_SHALLOW     TRUE

)
set(BENCHMARK_ENABLE_TESTING OFF)
FetchContent_MakeAvailable(googlebenchmark)
enable_testing()

#OpenMP
find_package(OpenMP)

add_executable(benchmarks src/benchmarks.cpp)
target_link_libraries(benchmarks PRIVATE benchmark::benchmark OpenMP::OpenMP_CXX)

add_executable(test_exe src/test.cpp)
target_link_libraries(test_exe PRIVATE OpenMP::OpenMP_CXX)
