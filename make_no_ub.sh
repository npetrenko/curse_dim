#!/bin/bash
mkdir no_ub
cd no_ub
g++ -I../include -Wall -Wextra -Wpedantic -Wno-c++98-compat -std=c++17 -fsanitize=address,undefined -fno-sanitize-recover=all -g -O0 ../ub_example/ub_example.cpp
