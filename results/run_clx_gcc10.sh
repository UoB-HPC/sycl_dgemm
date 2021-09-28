#!/bin/bash

module load cray-mvapich2_noslurm/2.3.5
#module load cuda11.2/toolkit/11.2.0
#module load llvm/11.0
module load boost/1.73.0/gcc-10.2

module load gcc/10.3.0

g++ --version

export PATH=$PATH:$PWD/hipsycl-scoped-gcc/bin
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$PWD/hipsycl-scoped-gcc/lib
export LIBRARY_PATH=$LIBRARY_PATH:$PWD/hipsycl-scoped-gcc/lib

#syclcc -O3 -std=c++17 --gcc-toolchain=/lustre/projects/bristol/modules/gcc/9.3.0 -march=native dgemm.cpp -o dgemm --hipsycl-cpu-cxx=/projects/bristol/modules/llvm/11.0/bin/clang++ --hipsycl-platform=cpu --hipsycl-targets=omp 

syclcc -O3 -std=c++17 -march=native dgemm.cpp -o dgemm --hipsycl-cpu-cxx=g++ --hipsycl-platform=cpu --hipsycl-targets=omp

export OMP_NUM_THREADS=40
export OMP_PLACES=cores
export OMP_PROC_BIND=true

runs=5

for i in $(seq 10 12); do
  N=$((2**$i))
  echo $N

  for j in $(seq 1 $runs); do
    ./dgemm $N $N $N
  done
done

