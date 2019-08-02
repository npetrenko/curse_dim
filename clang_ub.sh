#!/bin/bash
mkdir clang && cd clang
cmake -DCMAKE_BUILD_TYPE=ASAN -DCMAKE_CXX_COMPILER=clang++ ..
make main
./main
