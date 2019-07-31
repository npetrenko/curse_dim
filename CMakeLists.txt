cmake_minimum_required(VERSION 3.12 FATAL_ERROR)
project(curse_dim)

#set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-c++98-compat")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS} ${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native -mtune=native -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_TSAN "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-sanitize-recover=all -g -O0")
set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-sanitize-recover=all -g -O0")
set(CMAKE_CXX_FLAGS_RELASAN "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-sanitize-recover=all -g -O3 -march=native -mtune=native -fno-omit-frame-pointer")

find_package(glog 0.4.0 REQUIRED)
find_package(GTest REQUIRED)
add_subdirectory(thread_pool)
enable_testing()

set(SRC_DIR "src")
set(TEST_DIR "tests")
set(CPP_DIR "cpp")
set(MAIN_DIR "main")

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
target_include_directories(bellman PUBLIC .)
target_link_libraries(bellman glog::glog thread_pool)


set(TEST_SOURCES bellman.cpp storage.cpp type_traits.cpp uniform_operator_build.cpp)
prepend(TEST_SOURCES ${TEST_DIR} ${TEST_SOURCES})
add_executable(tests ${TEST_SOURCES})
target_link_libraries(tests bellman gtest_main gtest glog::glog)
add_test(tests tests)

set(MAIN_SOURCES main.cpp main_sources/pendulum.cpp main_sources/uniform_experiment.cpp main_sources/experiment.cpp)
prepend(MAIN_SOURCES ${MAIN_DIR} ${MAIN_SOURCES})
add_executable(main ${MAIN_SOURCES})
target_include_directories(main PUBLIC "${MAIN_DIR}/main_includes")
target_link_libraries(main bellman)
