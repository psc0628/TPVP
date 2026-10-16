cmake_minimum_required(VERSION 3.18)
project(cuda_visibility VERSION 1.0.0 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(octomap 1.10 REQUIRED)
find_package(Threads REQUIRED)

option(CUDA_VISIBILITY_ENABLE_CUDA "Build the CUDA backend" ON)
set(raycaster_source src/cuda_raycaster_stub.cpp)
if(CUDA_VISIBILITY_ENABLE_CUDA)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(raycaster_source src/cuda_raycaster.cu)
endif()

add_library(cuda_visibility
  src/camera_model.cpp
  src/visibility.cpp
  ${raycaster_source})
target_include_directories(cuda_visibility PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  ${OCTOMAP_INCLUDE_DIRS})
target_link_libraries(cuda_visibility PUBLIC
  Eigen3::Eigen octomap octomath Threads::Threads)

add_executable(visibility_example examples/visibility_example.cpp)
target_link_libraries(visibility_example PRIVATE cuda_visibility)

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(cuda_visibility PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-Wall;-Wextra;-Wpedantic>)
  target_compile_options(visibility_example PRIVATE -Wall -Wextra -Wpedantic)
endif()

if(BUILD_TESTING)
  add_executable(test_visibility test/test_visibility.cpp)
  target_link_libraries(test_visibility PRIVATE cuda_visibility)
  add_test(NAME visibility_cpu COMMAND test_visibility cpu)
  add_test(NAME example_cpu COMMAND visibility_example --backend cpu)
  add_test(NAME example_files COMMAND visibility_example --backend cpu
    --geometry ${CMAKE_CURRENT_SOURCE_DIR}/examples/geometry.xyz
    --views ${CMAKE_CURRENT_SOURCE_DIR}/examples/views.csv
    --camera ${CMAKE_CURRENT_SOURCE_DIR}/examples/camera.txt
    --precovered ${CMAKE_CURRENT_SOURCE_DIR}/examples/passive.xyz)
  if(CUDA_VISIBILITY_ENABLE_CUDA)
    add_test(NAME visibility_cpu_cuda_parity COMMAND test_visibility verify)
    add_test(NAME example_cpu_cuda_parity COMMAND visibility_example --backend verify)
  endif()
endif()

install(TARGETS cuda_visibility visibility_example
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
