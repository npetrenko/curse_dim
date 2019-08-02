cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(bellman)

#set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-c++98-compat")
set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-sanitize-recover=all -g -O0")

add_executable(ub_example "ub_example/ub_example.cpp")
target_include_directories(ub_example PUBLIC include)
