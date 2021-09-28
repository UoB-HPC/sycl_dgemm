#!/bin/bash

#module load cray-mvapich2_noslurm/2.3.5
#module unload cuda11.2/toolkit/11.2.0


source $HOME/intel/oneapi/setvars.sh
source $HOME/codes/everythingsreduced/src/dpcpp_compiler/startup.sh

clang++ -O3 -std=c++17 dgemm.cpp -o dgemm -fsycl -fsycl-unnamed-lambda


runs=5

for i in $(seq 10 12); do
  N=$((2**$i))
  echo $N

  for j in $(seq 1 $runs); do
    ./dgemm $N $N $N
  done
done


