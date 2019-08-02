#!/bin/bash
mkdir ub
cd ub
cmake -DCMAKE_BUILD_TYPE=ASAN -DCMAKE_CXX_COMPILER=g++ ..
make ub_example
./ub_example
