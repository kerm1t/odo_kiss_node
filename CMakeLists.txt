# ply_odom: relative pose between two PLY scans with KISS-ICP
#   cmake -B build && cmake --build build -j
#   ./build/ply_odom a.ply b.ply
cmake_minimum_required(VERSION 3.16...3.28)
project(ply_odom LANGUAGES CXX)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(FetchContent)

# KISS-ICP C++ core + pipeline. Uses system Eigen3 / Sophus / TBB / tsl-robin-map
# when found, otherwise fetches them itself.
FetchContent_Declare(kiss_icp
  GIT_REPOSITORY https://github.com/PRBonn/kiss-icp.git
  GIT_TAG        v1.3.0
  GIT_SHALLOW    ON
  SOURCE_SUBDIR  cpp/kiss_icp)

# happly: single header, no CMake project of its own
FetchContent_Declare(happly
  GIT_REPOSITORY https://github.com/nmwsharp/happly.git
  GIT_TAG        ed5480454d15e5287370eadc3682e9d40ba395e1)

FetchContent_MakeAvailable(kiss_icp happly)

add_executable(ply_odom main.cpp)
target_include_directories(ply_odom SYSTEM PRIVATE ${happly_SOURCE_DIR})
target_link_libraries(ply_odom PRIVATE kiss_icp_pipeline)
