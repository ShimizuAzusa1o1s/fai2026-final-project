#!/bin/bash
cd "$(dirname "$0")" || exit
cd ../core || exit
g++ -O3 -shared -fPIC fast_engine.cpp -o fast_engine.so -fopenmp
echo "Compiled fast_engine.so successfully."

