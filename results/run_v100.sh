#!/bin/bash

module load cray-mvapich2_noslurm/2.3.5
module load boost/1.73.0/gcc-10.2
module load gcc/9.3.0
module load cuda11.2/toolkit/11.2.0

export PATH=$PWD/llvm-install/bin:$PATH:$PWD/hipsycl-scoped/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/llvm-install/lib
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/llvm-install/lib:$PWD/hipsycl-scoped/lib
#export LIBRARY_PATH=$LIBRARY_PATH:$PWD/llvm-install/lib:$PWD/hipsycl-scoped/lib

rm -f dgemm_gpu
syclcc -g -O3 -std=c++17 --gcc-toolchain=/lustre/projects/bristol/modules/gcc/9.3.0 dgemm.cpp -o dgemm_gpu --hipsycl-targets="cuda:sm_70" --cuda-path=/cm/shared/apps/cuda11.2/toolkit/11.2.0


export HIPSYCL_VISIBILITY_MASK=cuda
for i in $(seq 10 12); do
  N=$((2**$i))
  echo $N

  for j in {1..5}; do
    LD_PRELOAD=/cm/local/apps/cuda/libs/current/lib64/libcuda.so.1:/lustre/projects/bristol/modules/gcc/9.3.0/lib64/libstdc++.so ./dgemm_gpu $N $N $N
  done
done



