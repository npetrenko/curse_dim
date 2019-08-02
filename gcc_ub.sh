#!/bin/bash
mkdir gcc && cd gcc
cmake -DCMAKE_BUILD_TYPE=ASAN -DCMAKE_CXX_COMPILER=g++ ..
make bellman_tests
./bellman/bellman_tests --gtest_filter=UB*
