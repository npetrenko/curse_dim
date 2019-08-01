cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(thread_pool)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "-Wall -Wextra -Wpedantic -Wno-c++98-compat")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS} -g -fno-omit-frame-pointer -O0")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} -O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELEASE} -g -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_TSAN "${CMAKE_CXX_FLAGS} -fsanitize=thread -g -O0")
set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -g -O0")

find_package(benchmark REQUIRED)

set(SRC_DIR "cpp")
set(BM_DIR "benchmarks")

file(GLOB CPPS "${SRC_DIR}/*.cpp")

add_library(thread_pool ${CPPS})
target_include_directories(thread_pool PUBLIC include)
target_link_libraries(thread_pool pthread)

add_executable(pool_bm "${BM_DIR}/benchmark.cpp")
target_link_libraries(pool_bm benchmark thread_pool)
