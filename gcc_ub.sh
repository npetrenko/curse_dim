#!/bin/bash
mkdir gcc && cd gcc
cmake -DCMAKE_BUILD_TYPE=ASAN -DCMAKE_CXX_COMPILER=g++ ..
make ub_example
./bellman/ub_example
