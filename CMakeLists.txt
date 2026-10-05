# ply_odom: KISS-ICP odometry
#   ply_odom        PLY folder -> trajectory file, optional SDL2 + OpenGL 3.3 viewer
#   kiss_odom_node  eCAL 5 node: foxglove.PointCloud in -> foxglove.Odometry out
#   ply_pub         replays a PLY folder as foxglove.PointCloud (test input for the node)
#
#   cmake -B build && cmake --build build -j
#   ./build/ply_odom <dir | a.ply b.ply ...> [-o traj.txt] [-r max_range_m] [-v voxel_m] [-p pts_per_voxel] [-i max_icp_iter] [-d vis_voxel_m] [-n]
#   ./build/kiss_odom_node [--in pointcloud] [--out odometry] [--odom-frame odom] [--body-frame <id>] [-r ..] [-v ..] [-p ..] [-i ..] [-q]
#   ./build/ply_pub <dir> [--out pointcloud] [--hz 10] [--frame lidar] [--loop]
cmake_minimum_required(VERSION 3.16...3.28)
project(ply_odom LANGUAGES CXX)

option(WITH_VIEWER "Build the SDL2/OpenGL viewer" ON)
option(WITH_ECAL_NODE "Build the eCAL 5 node and the PLY replay publisher" ON)

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

# eCAL 5 node. Needs eCAL 5 and protobuf (protoc); skipped when either is missing.
# The foxglove .proto files are vendored in proto/ (from foxglove-sdk, MIT). To build
# against the stack's own copies instead: -DFOXGLOVE_PROTO_DIR=<dir containing foxglove/>.
if(WITH_ECAL_NODE)
  find_package(eCAL QUIET)
  find_package(Protobuf QUIET)
  if(eCAL_FOUND AND Protobuf_FOUND)
    set(FOXGLOVE_PROTO_DIR ${CMAKE_CURRENT_SOURCE_DIR}/proto CACHE PATH
        "Directory that contains foxglove/*.proto")
    set(fg_names Odometry PointCloud PackedElementField Pose Quaternion Vector3 KeyValuePair)

    # foxglove/Odometry.proto labels two message fields `optional`. protoc < 3.15 rejects
    # that (3.11, as bundled with eCAL 5.13: "Explicit 'optional' labels are disallowed";
    # 3.12-3.14: "--experimental_allow_proto3_optional was not set"). So compile copies
    # with the label removed. For message-typed fields the label changes nothing: same
    # wire format, and has_...() exists either way.
    # OFF = compile the .proto files unmodified (needs protoc >= 3.15).
    option(FOXGLOVE_STRIP_PROTO3_OPTIONAL "Remove proto3 'optional' labels before running protoc" ON)
    set(fg_proto_root ${FOXGLOVE_PROTO_DIR})
    if(FOXGLOVE_STRIP_PROTO3_OPTIONAL)
      set(fg_proto_root ${CMAKE_CURRENT_BINARY_DIR}/proto_compat)
      foreach(p ${fg_names})
        set(src ${FOXGLOVE_PROTO_DIR}/foxglove/${p}.proto)
        set(dst ${fg_proto_root}/foxglove/${p}.proto)
        file(READ ${src} txt)
        string(REGEX REPLACE "(^|\n)([ \t]*)optional[ \t]+" "\\1\\2" txt "${txt}")
        set(old "")
        if(EXISTS ${dst})
          file(READ ${dst} old)
        endif()
        if(NOT "${old}" STREQUAL "${txt}") # leave the timestamp alone if nothing changed
          file(WRITE ${dst} "${txt}")
        endif()
        set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${src})
      endforeach()
    endif()

    set(fg_protos)
    foreach(p ${fg_names})
      list(APPEND fg_protos ${fg_proto_root}/foxglove/${p}.proto)
    endforeach()
    add_library(foxglove_proto STATIC)
    PROTOBUF_TARGET_CPP(foxglove_proto ${fg_proto_root} ${fg_protos}) # from eCAL's CMake helpers
    target_link_libraries(foxglove_proto PUBLIC protobuf::libprotobuf)

    add_executable(kiss_odom_node odom_node.cpp)
    target_link_libraries(kiss_odom_node PRIVATE kiss_icp_pipeline foxglove_proto eCAL::core
                                                 Threads::Threads)

    if(NOT TARGET Eigen3::Eigen) # KISS-ICP found the system one inside its own scope
      find_package(Eigen3 REQUIRED NO_MODULE)
    endif()
    add_executable(ply_pub ply_pub.cpp)
    target_include_directories(ply_pub SYSTEM PRIVATE ${happly_SOURCE_DIR})
    target_link_libraries(ply_pub PRIVATE Eigen3::Eigen foxglove_proto eCAL::core Threads::Threads)
    message(STATUS "ply_odom: eCAL node enabled (eCAL ${eCAL_VERSION})")
  else()
    message(STATUS "ply_odom: eCAL or Protobuf not found, building without the eCAL node")
  endif()
endif()
