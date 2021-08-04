CXX = dpcpp
FLAGS = -fsycl

dgemm: dgemm.cpp
	$(CXX) -std=c++14 -O3 $(FLAGS) $^ -o $@

