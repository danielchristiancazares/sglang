cmake_minimum_required(VERSION 3.24)
project(sglang_native_diffusiongemma LANGUAGES CXX)

if(NOT WIN32 OR NOT MSVC)
    message(FATAL_ERROR "This native-Windows trial requires MSVC and CPython 3.13.")
endif()

get_filename_component(SGLANG_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../.." ABSOLUTE)
find_package(Python3 3.13 EXACT REQUIRED COMPONENTS Interpreter Development.Embed)
get_filename_component(DIFFUSION_PYTHON_BASE "${Python3_STDLIB}" DIRECTORY)
set(DIFFUSION_PYBIND_INCLUDE "${SGLANG_ROOT}/.venv/Lib/site-packages/torch/include")
if(NOT EXISTS "${DIFFUSION_PYBIND_INCLUDE}/pybind11/embed.h")
    message(FATAL_ERROR "The checkout's PyTorch installation must provide pybind11 headers.")
endif()

add_library(diffusiongemma_runtime SHARED runtime.cpp)
target_compile_features(diffusiongemma_runtime PRIVATE cxx_std_20)
target_compile_options(diffusiongemma_runtime PRIVATE /W4 /EHsc /utf-8)
target_compile_definitions(diffusiongemma_runtime PRIVATE NOMINMAX
    DG_REPO_ROOT="${SGLANG_ROOT}" DG_PYTHON_BASE="${DIFFUSION_PYTHON_BASE}")
target_include_directories(diffusiongemma_runtime SYSTEM PRIVATE "${DIFFUSION_PYBIND_INCLUDE}")
target_link_libraries(diffusiongemma_runtime PRIVATE Python3::Python)

add_executable(diffusiongemma launcher.cpp)
target_compile_features(diffusiongemma PRIVATE cxx_std_20)
target_compile_options(diffusiongemma PRIVATE /W4 /EHsc /utf-8)
target_compile_definitions(diffusiongemma PRIVATE NOMINMAX
    DG_PYTHON_BASE=L"${DIFFUSION_PYTHON_BASE}")
add_dependencies(diffusiongemma diffusiongemma_runtime)

set_target_properties(diffusiongemma diffusiongemma_runtime PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
