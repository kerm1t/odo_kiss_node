cmake_minimum_required(VERSION 3.16)
project(odo_kiss CXX)

# KISS-ICP benötigt zwingend C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# --- 1. Abhängigkeiten von KISS-ICP im System finden ---
find_package(Eigen3 REQUIRED)
find_package(TBB REQUIRED) # Für Parallelisierung

# --- 2. KISS-ICP als Subprojekt hinzufügen ---
# Wir weisen CMake an, direkt in den cpp-Ordner von KISS-ICP zu schauen
add_subdirectory(thirdparty/kiss_icp ${CMAKE_CURRENT_BINARY_DIR}/kiss_icp)

include_directories(thirdparty)

# --- 3. Eigenes Executable erstellen ---
add_executable(mein_programm src/main.cpp)

# --- 4. Mit KISS-ICP verlinken ---
# Das Target 'kiss_icp::kiss_icp' wird vom Subprojekt bereitgestellt
# target_link_libraries(mein_programm PRIVATE kiss_icp::kiss_icp)
#target_link_libraries(mein_programm PRIVATE kiss_icp Eigen3::Eigen sophus)
target_link_libraries(mein_programm PRIVATE kiss_icp_pipeline Eigen3::Eigen sophus)
