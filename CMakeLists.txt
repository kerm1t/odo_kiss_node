# ply_odom: KISS-ICP odometry over a sequence of PLY scans -> trajectory file,
# with an optional SDL2 + OpenGL 3.3 viewer (accumulated cloud + ego trajectory)
#   cmake -B build && cmake --build build -j
#   ./build/ply_odom <dir | a.ply b.ply ...> [-o traj.txt] [-r max_range_m] [-v voxel_m] [-d vis_voxel_m] [-n]
cmake_minimum_required(VERSION 3.16...3.28)
project(ply_odom LANGUAGES CXX)

option(WITH_VIEWER "Build the SDL2/OpenGL viewer" ON)

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

find_package(Threads REQUIRED)

add_executable(ply_odom main.cpp)
target_include_directories(ply_odom SYSTEM PRIVATE ${happly_SOURCE_DIR})
target_link_libraries(ply_odom PRIVATE kiss_icp_pipeline Threads::Threads)

# Viewer: system SDL2 + OpenGL (Linux: libsdl2-dev libgl-dev, macOS: brew install sdl2).
# Falls back to a headless build when either is missing.
if(WITH_VIEWER)
  find_package(SDL2 QUIET)
  find_package(OpenGL QUIET)
  if(SDL2_FOUND AND OpenGL_FOUND)
    target_compile_definitions(ply_odom PRIVATE PLY_ODOM_VIEWER)
    if(TARGET SDL2::SDL2)
      target_link_libraries(ply_odom PRIVATE SDL2::SDL2)
    else() # older SDL2 packages only export variables
      target_include_directories(ply_odom SYSTEM PRIVATE ${SDL2_INCLUDE_DIRS})
      target_link_libraries(ply_odom PRIVATE ${SDL2_LIBRARIES})
    endif()
    target_link_libraries(ply_odom PRIVATE OpenGL::GL)
    message(STATUS "ply_odom: viewer enabled")
  else()
    message(STATUS "ply_odom: SDL2 or OpenGL not found, building without viewer")
  endif()
endif()
