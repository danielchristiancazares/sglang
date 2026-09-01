cmake_minimum_required(VERSION 3.26 FATAL_ERROR)

project(sglang-native LANGUAGES CXX CUDA)

if(NOT WIN32 OR NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "sglang-native currently requires Windows x64")
endif()
if(NOT MSVC)
  message(FATAL_ERROR "sglang-native currently requires the MSVC toolchain")
endif()

find_package(CUDAToolkit 13.3 REQUIRED)
if(CUDAToolkit_VERSION VERSION_LESS 13.3 OR
   NOT CUDAToolkit_VERSION VERSION_LESS 13.4)
  message(
    FATAL_ERROR
      "sglang-native requires the qualified CUDA 13.3 toolkit; found ${CUDAToolkit_VERSION}")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_ARCHITECTURES 120)
set(CMAKE_CUDA_RUNTIME_LIBRARY Shared)

function(sglang_native_target target)
  target_compile_options(
    ${target}
    PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/EHsc;/W4;/WX;/permissive-;/Zc:__cplusplus>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/EHsc;-Xcompiler=/W4;-Xcompiler=/WX;-Xcompiler=/permissive-;-Xcompiler=/Zc:__cplusplus;-Xcompiler=/wd4211;--threads=2>)
  set_target_properties(
    ${target}
    PROPERTIES
      CUDA_ARCHITECTURES 120
      CUDA_SEPARABLE_COMPILATION OFF)
endfunction()

add_library(sglang_native_tensor STATIC src/tensor_view.cpp)
target_include_directories(
  sglang_native_tensor
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
sglang_native_target(sglang_native_tensor)

add_library(sglang_native_cuda_runtime STATIC src/cuda_graph_resources.cpp)
target_include_directories(
  sglang_native_cuda_runtime
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(
  sglang_native_cuda_runtime
  PUBLIC sglang_native_tensor CUDA::cudart)
sglang_native_target(sglang_native_cuda_runtime)

add_library(
  sglang_native_kernels
  STATIC
    src/linear_rejection_sampling.cpp
    src/linear_rejection_sampling.cu
    src/linear_verify_rng.cpp
    src/linear_verify_rng.cu)
target_include_directories(
  sglang_native_kernels
  PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>)
target_link_libraries(
  sglang_native_kernels
  PUBLIC sglang_native_cuda_runtime CUDA::cudart)
sglang_native_target(sglang_native_kernels)

add_library(sglang_native INTERFACE)
target_link_libraries(sglang_native INTERFACE sglang_native_kernels)

add_executable(sglang_native_probe src/native_probe.cpp)
target_link_libraries(sglang_native_probe PRIVATE sglang_native)
sglang_native_target(sglang_native_probe)

include(CTest)
if(BUILD_TESTING)
  add_executable(
    sglang_native_tensor_test
    test/tensor_view_test.cpp)
  target_link_libraries(
    sglang_native_tensor_test PRIVATE sglang_native_tensor)
  sglang_native_target(sglang_native_tensor_test)

  add_executable(
    sglang_native_tensor_pack1_test
    src/tensor_view.cpp
    test/tensor_view_test.cpp)
  target_include_directories(
    sglang_native_tensor_pack1_test
    PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include)
  target_compile_options(
    sglang_native_tensor_pack1_test PRIVATE $<$<COMPILE_LANGUAGE:CXX>:/Zp1>)
  sglang_native_target(sglang_native_tensor_pack1_test)

  add_library(
    sglang_native_tensor_cuda_compile_probe
    OBJECT test/tensor_view_cuda_compile_test.cu)
  target_link_libraries(
    sglang_native_tensor_cuda_compile_probe PRIVATE sglang_native_tensor)
  sglang_native_target(sglang_native_tensor_cuda_compile_probe)

  add_executable(
    sglang_native_cuda_runtime_host_test
    test/cuda_graph_resources_host_test.cpp)
  target_link_libraries(
    sglang_native_cuda_runtime_host_test
    PRIVATE sglang_native_cuda_runtime)
  sglang_native_target(sglang_native_cuda_runtime_host_test)

  add_executable(
    sglang_native_linear_rejection_host_test
    test/linear_rejection_sampling_host_test.cpp)
  target_link_libraries(
    sglang_native_linear_rejection_host_test
    PRIVATE sglang_native_kernels)
  sglang_native_target(sglang_native_linear_rejection_host_test)

  add_executable(
    sglang_native_linear_verify_rng_host_test
    test/linear_verify_rng_host_test.cpp)
  target_link_libraries(
    sglang_native_linear_verify_rng_host_test
    PRIVATE sglang_native_kernels)
  sglang_native_target(sglang_native_linear_verify_rng_host_test)

  add_executable(
    sglang_native_cuda_runtime_test
    test/cuda_graph_resources_test.cu)
  target_link_libraries(
    sglang_native_cuda_runtime_test
    PRIVATE sglang_native_cuda_runtime)
  sglang_native_target(sglang_native_cuda_runtime_test)

  add_executable(
    sglang_native_linear_rejection_test
    test/linear_rejection_sampling_test.cu)
  target_link_libraries(
    sglang_native_linear_rejection_test
    PRIVATE sglang_native_kernels)
  sglang_native_target(sglang_native_linear_rejection_test)

  add_executable(
    sglang_native_linear_verify_rng_test
    test/linear_verify_rng_test.cu)
  target_link_libraries(
    sglang_native_linear_verify_rng_test
    PRIVATE sglang_native_kernels)
  sglang_native_target(sglang_native_linear_verify_rng_test)

  add_test(NAME native.tensor COMMAND sglang_native_tensor_test)
  add_test(NAME native.tensor_pack1 COMMAND sglang_native_tensor_pack1_test)
  add_test(
    NAME native.cuda_runtime_host
    COMMAND sglang_native_cuda_runtime_host_test)
  add_test(
    NAME native.linear_rejection_host
    COMMAND sglang_native_linear_rejection_host_test)
  add_test(
    NAME native.linear_verify_rng_host
    COMMAND sglang_native_linear_verify_rng_host_test)
  add_test(
    NAME native.cuda_runtime
    COMMAND sglang_native_cuda_runtime_test)
  add_test(
    NAME native.linear_rejection
    COMMAND sglang_native_linear_rejection_test)
  add_test(
    NAME native.linear_verify_rng
    COMMAND sglang_native_linear_verify_rng_test)
  add_test(NAME native.probe COMMAND sglang_native_probe)

  set_tests_properties(
    native.tensor
    native.tensor_pack1
    native.cuda_runtime_host
    native.linear_rejection_host
    native.linear_verify_rng_host
    native.probe
    PROPERTIES LABELS host)
  set_tests_properties(
    native.cuda_runtime
    native.linear_rejection
    native.linear_verify_rng
    PROPERTIES
      LABELS "cuda;sm120"
      RESOURCE_LOCK sglang_native_cuda
      RUN_SERIAL TRUE)

  find_program(
    SGLANG_NATIVE_COMPUTE_SANITIZER
    NAMES compute-sanitizer
    HINTS
      "${CUDAToolkit_BIN_DIR}"
      "${CUDAToolkit_ROOT}/compute-sanitizer"
      "${CUDAToolkit_LIBRARY_ROOT}/compute-sanitizer"
      "$ENV{CUDA_HOME}/compute-sanitizer")
  if(SGLANG_NATIVE_COMPUTE_SANITIZER)
    add_custom_target(
      sglang_native_cuda_runtime_memcheck
      COMMAND
        "${SGLANG_NATIVE_COMPUTE_SANITIZER}" --tool memcheck
        --error-exitcode 99 $<TARGET_FILE:sglang_native_cuda_runtime_test>
      DEPENDS sglang_native_cuda_runtime_test
      USES_TERMINAL)
    add_custom_target(
      sglang_native_linear_rejection_memcheck
      COMMAND
        "${SGLANG_NATIVE_COMPUTE_SANITIZER}" --tool memcheck
        --error-exitcode 99 $<TARGET_FILE:sglang_native_linear_rejection_test>
      DEPENDS sglang_native_linear_rejection_test
      USES_TERMINAL)
    add_custom_target(
      sglang_native_linear_verify_rng_memcheck
      COMMAND
        "${SGLANG_NATIVE_COMPUTE_SANITIZER}" --tool memcheck
        --error-exitcode 99 $<TARGET_FILE:sglang_native_linear_verify_rng_test>
      DEPENDS sglang_native_linear_verify_rng_test
      USES_TERMINAL)
  endif()
endif()
