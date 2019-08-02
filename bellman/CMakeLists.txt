cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(bellman)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-c++98-compat")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native -mtune=native -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_TSAN "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-sanitize-recover=all -g -O0")
set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS} -Wodr -fsanitize=address,undefined -fno-sanitize-recover=all -g -O0")
set(CMAKE_CXX_FLAGS_RELASAN "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-sanitize-recover=all -g -O3 -march=native -mtune=native -fno-omit-frame-pointer")

find_package(glog 0.4.0 REQUIRED)
find_package(GTest REQUIRED)
add_subdirectory(thread_pool)
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

set(SOURCES particle.cpp
  bellman_operators/abstract_bellman.cpp
  bellman_operators/environment.cpp
  bellman_operators/qfunc.cpp
  bellman_operators/stationary_operator.cpp
  bellman_operators/uniform_operator.cpp
  density_estimators/stationary_estimator.cpp)
prepend(SOURCES ${CPP_DIR} ${SOURCES})

add_library(bellman ${SOURCES})
target_include_directories(bellman PUBLIC include)
target_link_libraries(bellman glog::glog thread_pool)

set(TEST_SOURCES main.cpp bellman.cpp storage.cpp type_traits.cpp ub_example.cpp)
prepend(TEST_SOURCES ${TEST_DIR} ${TEST_SOURCES})
add_executable(bellman_tests ${TEST_SOURCES})
target_link_libraries(bellman_tests bellman gtest glog::glog)
add_test(bellman_tests bellman_tests)
set_target_properties(bellman_tests PROPERTIES EXCLUDE_FROM_ALL 1)
