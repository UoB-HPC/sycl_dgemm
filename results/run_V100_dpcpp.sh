#!/bin/bash

export DPCPP_HOME=$PWD/sycl_workspace
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$DPCPP_HOME/llvm/build/lib

rm -f dgemm_gpu

$DPCPP_HOME/llvm/build/bin/clang++ -std=c++17 -O3 -fsycl -fsycl-targets=nvptx64-nvidia-cuda dgemm.cpp --cuda-path=/cm/shared/apps/cuda10.2/toolkit/10.2.89 -o dgemm_gpu


for i in $(seq 10 12); do
  N=$((2**$i))
  echo $N

  for j in {1..5}; do
    ./dgemm_gpu $N $N $N
  done
done



