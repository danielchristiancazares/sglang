cmake_minimum_required(VERSION 3.24)
project(sglang_windows_ttft_launcher LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

# Declare first so the independent benchmark project does not recurse here.
add_library(sglang_windows_ttft src/launcher.cpp src/platform.cpp)
target_include_directories(sglang_windows_ttft PUBLIC include)
if(NOT TARGET sglang_benchmark)
  add_subdirectory(../../benchmark/native benchmark_native)
endif()
target_link_libraries(sglang_windows_ttft PUBLIC sglang_benchmark)
if(WIN32)
  target_compile_definitions(sglang_windows_ttft PRIVATE _WIN32_WINNT=0x0A00)
endif()

add_executable(serve_windows_ttft main.cpp)
target_link_libraries(serve_windows_ttft PRIVATE sglang_windows_ttft)

if(BUILD_TESTING)
  add_executable(windows_ttft_test test/launcher_test.cpp)
  target_link_libraries(windows_ttft_test PRIVATE sglang_windows_ttft)
  add_test(NAME windows_ttft COMMAND windows_ttft_test)
  set_tests_properties(windows_ttft PROPERTIES
    WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" TIMEOUT 60)
endif()
