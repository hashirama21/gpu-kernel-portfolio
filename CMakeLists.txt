cmake_minimum_required(VERSION 3.18)
project(gpu_kernel_portfolio LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release)
endif()

option(PORTFOLIO_NATIVE "Compile CPU code with -march=native (fastest here, not portable)" ON)
option(PORTFOLIO_OPENMP "Use OpenMP where applicable (projects 1, 2)" ON)

find_package(OpenMP QUIET)

# ---------------------------------------------------------------------------
# Common CPU-side helper (timer.hpp). Header-only.
# ---------------------------------------------------------------------------
add_library(portfolio_common INTERFACE)
target_include_directories(portfolio_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/common/include)
target_compile_options(portfolio_common INTERFACE -Wall -Wextra -Wpedantic)
if(PORTFOLIO_NATIVE)
  target_compile_options(portfolio_common INTERFACE -march=native)
endif()

enable_testing()

# ---------------------------------------------------------------------------
# Project 1: Vector Add (CPU always; CUDA if available)
# ---------------------------------------------------------------------------
add_executable(01_vector_add_cpu 01_vector_add/src/main_cpu.cpp)
target_include_directories(01_vector_add_cpu PRIVATE 01_vector_add/include)
target_link_libraries(01_vector_add_cpu PRIVATE portfolio_common)
if(PORTFOLIO_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(01_vector_add_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(01_vector_add_test_cpu 01_vector_add/tests/test_vector_add_cpu.cpp)
target_include_directories(01_vector_add_test_cpu PRIVATE 01_vector_add/include)
target_link_libraries(01_vector_add_test_cpu PRIVATE portfolio_common)
if(PORTFOLIO_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(01_vector_add_test_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME 01_vector_add_cpu COMMAND 01_vector_add_test_cpu)

add_executable(01_vector_add_indexing_sim 01_vector_add/tests/test_vector_add_cuda_indexing_sim.cpp)
target_link_libraries(01_vector_add_indexing_sim PRIVATE portfolio_common)
add_test(NAME 01_vector_add_indexing_sim COMMAND 01_vector_add_indexing_sim)

# ---------------------------------------------------------------------------
# Project 2: Matrix Multiplication (CPU only)
# ---------------------------------------------------------------------------
add_executable(02_matmul_cpu 02_matmul_cpu/src/main.cpp)
target_include_directories(02_matmul_cpu PRIVATE 02_matmul_cpu/include)
target_link_libraries(02_matmul_cpu PRIVATE portfolio_common)
if(PORTFOLIO_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(02_matmul_cpu PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(02_matmul_test 02_matmul_cpu/tests/test_matmul.cpp)
target_include_directories(02_matmul_test PRIVATE 02_matmul_cpu/include)
target_link_libraries(02_matmul_test PRIVATE portfolio_common)
if(PORTFOLIO_OPENMP AND OpenMP_CXX_FOUND)
  target_link_libraries(02_matmul_test PRIVATE OpenMP::OpenMP_CXX)
endif()
add_test(NAME 02_matmul COMMAND 02_matmul_test)

# ---------------------------------------------------------------------------
# Project 3: TensorView (CPU only)
# ---------------------------------------------------------------------------
add_executable(03_tensor_view 03_tensor_view/src/main.cpp)
target_include_directories(03_tensor_view PRIVATE 03_tensor_view/include)
target_link_libraries(03_tensor_view PRIVATE portfolio_common)

add_executable(03_tensor_view_test 03_tensor_view/tests/test_tensor_view.cpp)
target_include_directories(03_tensor_view_test PRIVATE 03_tensor_view/include)
target_link_libraries(03_tensor_view_test PRIVATE portfolio_common)
add_test(NAME 03_tensor_view COMMAND 03_tensor_view_test)

# ---------------------------------------------------------------------------
# Projects 4-6: CPU-side algorithm simulations always built (no GPU needed).
# ---------------------------------------------------------------------------
add_executable(04_reduction_sim 04_reduction/tests/test_reduction_cpu_sim.cpp)
target_link_libraries(04_reduction_sim PRIVATE portfolio_common)
add_test(NAME 04_reduction_sim COMMAND 04_reduction_sim)

add_executable(05_histogram_sim 05_histogram/tests/test_histogram_cpu_sim.cpp)
target_link_libraries(05_histogram_sim PRIVATE portfolio_common)
add_test(NAME 05_histogram_sim COMMAND 05_histogram_sim)

add_executable(06_conv2d_sim 06_convolution/tests/test_conv2d_cpu_sim.cpp)
target_include_directories(06_conv2d_sim PRIVATE 06_convolution/include)
target_link_libraries(06_conv2d_sim PRIVATE portfolio_common)
add_test(NAME 06_conv2d_sim COMMAND 06_conv2d_sim)

# ---------------------------------------------------------------------------
# CUDA targets: only if a CUDA compiler is actually found. Building this portfolio without a GPU (as in
# the environment it was written in) still builds and runs every CPU target and every CPU-side
# algorithm simulation above; it simply skips the .cu targets below instead of failing the whole build.
# ---------------------------------------------------------------------------
include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)  # nvcc's C++23 device-code support varies by toolkit version; host code
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)  # still gets -std=c++23 host-side via CMAKE_CXX_STANDARD above.
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75 80 86 89)  # T4, A100, RTX 30xx, RTX 40xx: broad, safe default
  endif()

  add_executable(01_vector_add_gpu 01_vector_add/src/main.cu)
  target_include_directories(01_vector_add_gpu PRIVATE 01_vector_add/include common/include)

  add_executable(04_reduction_gpu 04_reduction/src/main.cu)
  target_include_directories(04_reduction_gpu PRIVATE 04_reduction/include common/include)

  add_executable(05_histogram_gpu 05_histogram/src/main.cu)
  target_include_directories(05_histogram_gpu PRIVATE 05_histogram/include common/include)

  add_executable(06_conv2d_gpu 06_convolution/src/main.cu)
  target_include_directories(06_conv2d_gpu PRIVATE 06_convolution/include common/include)

  message(STATUS "CUDA compiler found (${CMAKE_CUDA_COMPILER}): building GPU targets "
                  "01_vector_add_gpu, 04_reduction_gpu, 05_histogram_gpu, 06_conv2d_gpu.")
else()
  message(STATUS "No CUDA compiler found: skipping GPU targets. CPU targets and CPU-side algorithm "
                  "simulations (which cover projects 1, 4, 5, and 6's kernel logic without a GPU) are "
                  "still built and tested normally. See README.md.")
endif()
