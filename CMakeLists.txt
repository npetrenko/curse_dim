cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(curse_dim)

set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-c++98-compat")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "${CMAKE_CXX_FLAGS_RELWITHDEBINFO} -march=native -mtune=native -fno-omit-frame-pointer")
set(CMAKE_CXX_FLAGS_TSAN "${CMAKE_CXX_FLAGS} -fsanitize=thread -fno-sanitize-recover=all -g -O1")
set(CMAKE_CXX_FLAGS_ASAN "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined -fno-sanitize-recover=all -g -O1")

find_package(glog 0.4.0 REQUIRED)
find_package(GTest REQUIRED)
add_subdirectory(thread_pool)
enable_testing()

set(SRC_DIR "src")
set(TEST_DIR "tests")
set(CPP_DIR "cpp")
set(MAIN_DIR "main")

add_library(bellman "${CPP_DIR}/particle.cpp")
target_include_directories(bellman PUBLIC .)
target_link_libraries(bellman glog::glog thread_pool)

add_executable(tests "${TEST_DIR}/bellman.cpp" "${TEST_DIR}/storage.cpp")
target_link_libraries(tests bellman gtest_main gtest glog::glog)
add_test(tests tests)

add_executable(main "${MAIN_DIR}/main.cpp")
target_link_libraries(main bellman)
