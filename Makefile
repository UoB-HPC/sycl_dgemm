dgemm: dgemm.cpp
	$(CXX) -std=c++14 -O3 $^ -o $@

