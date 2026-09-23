cmake_minimum_required(VERSION 3.24)
project(sglang_native_benchmarks LANGUAGES CXX)
set(CMAKE_CXX_EXTENSIONS OFF)

include(CTest)
find_package(Threads REQUIRED)

if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
endif()

add_library(sglang_benchmark
  src/arguments.cpp
  src/http_client.cpp
  src/json.cpp
  src/openai_benchmark.cpp
  src/sha256.cpp
  src/sse_parser.cpp
  src/ttft_benchmark.cpp)
target_include_directories(sglang_benchmark PUBLIC include)
target_compile_features(sglang_benchmark PUBLIC cxx_std_23)
set_target_properties(sglang_benchmark PROPERTIES CXX_EXTENSIONS OFF)

# Propagate the language/diagnostic contract to clients and their host tests.
if(MSVC)
  target_compile_options(sglang_benchmark PUBLIC
    /std:c++latest /Zc:__cplusplus /EHsc /W4 /WX /permissive- /utf-8)
else()
  target_compile_options(sglang_benchmark PUBLIC
    -Wall -Wextra -Wpedantic -Werror)
endif()
if(WIN32)
  target_compile_definitions(sglang_benchmark PUBLIC NOMINMAX WIN32_LEAN_AND_MEAN)
  target_link_libraries(sglang_benchmark PUBLIC ws2_32)
endif()
target_link_libraries(sglang_benchmark PUBLIC Threads::Threads)

foreach(client bench_openai_stream bench_spec_acceptance bench_ttft)
  add_executable(${client} ${client}.cpp)
  target_link_libraries(${client} PRIVATE sglang_benchmark)
endforeach()

if(BUILD_TESTING)
  foreach(suite arguments http_sse json_sha256 openai_benchmark ttft_benchmark)
    add_executable(${suite}_test test/${suite}_test.cpp)
    target_link_libraries(${suite}_test PRIVATE sglang_benchmark)
    add_test(NAME ${suite} COMMAND ${suite}_test)
    set_tests_properties(${suite} PROPERTIES
      WORKING_DIRECTORY "${CMAKE_BINARY_DIR}" TIMEOUT 60)
  endforeach()
endif()
