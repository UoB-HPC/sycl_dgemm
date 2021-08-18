# DPCPP
CXX = dpcpp
FLAGS = -fsycl

# HIPSYCL
CXX = syclcc
FLAGS = --hipsycl-targets=omp

dgemm: dgemm.cpp
	$(CXX) -std=c++17 -O3 $(FLAGS) $^ -o $@

