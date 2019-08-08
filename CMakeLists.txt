cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(curse_dim)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-c++98-compat")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native -mtune=native -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_TSAN "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-sanitize-recover=all -g -O0")
set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS} -Wodr -fsanitize=address,undefined -fno-sanitize=vptr -fno-sanitize-recover=all -g -O0")
set(CMAKE_CXX_FLAGS_RELASAN "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-sanitize=vptr -fno-sanitize-recover=all -g -O3 -march=native -mtune=native -fno-omit-frame-pointer")

find_package(glog 0.4.0 REQUIRED)
find_package(GTest REQUIRED)
find_package(gflags REQUIRED)
add_subdirectory(bellman)
enable_testing()

set(TEST_DIR "tests")
set(CPP_DIR "cpp")

function(prepend var prefix)
  set(listVar "")
  foreach(f ${ARGN})
    list(APPEND listVar "${prefix}/${f}")
  endforeach(f)
  set(${var} ${listVar} PARENT_SCOPE)
endfunction()

set(SOURCES pendulum.cpp
  uniform_experiment.cpp
  stationary_experiment.cpp
  random_experiment.cpp
  experiment.cpp
  experiment_printer.cpp)

prepend(SOURCES ${CPP_DIR} ${SOURCES})
add_library(curse_dim ${SOURCES})
target_include_directories(curse_dim PUBLIC include)
target_link_libraries(curse_dim bellman)

set(TEST_SOURCES pendulum_kernel.cpp main.cpp)
prepend(TEST_SOURCES ${TEST_DIR} ${TEST_SOURCES})
add_executable(curse_tests ${TEST_SOURCES})
target_link_libraries(curse_tests curse_dim gtest glog::glog)
add_test(curse_tests curse_tests)
set_target_properties(curse_tests PROPERTIES EXCLUDE_FROM_ALL 1)

add_executable(main "${CPP_DIR}/main.cpp")
target_link_libraries(main curse_dim gflags)
