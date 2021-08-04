// Copyright (c) 2021 SYCL DGEMM Benchmark authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <chrono>

//#include <sycl/sycl.hpp>

// Matrix multiplication benchmark
//
//  Computes C = A x B for matrices A, B and C of sizes:
//  A is N x P
//  B is P x M
//  C is N x M
//

const double Aval = 3.0;
const double Bval = 5.0;

void init_input_matrices(const int Ndim, const int Mdim, const int Pdim, double *A, double *B);
void zero_matrix(const int N, const int M, double * C);
void matmul(const int Ndim, const int Mdim, const int Pdim, double *A, double *B, double *C);
void get_true_solution(const int Ndim, const int Mdim, const int Pdim, double * C);
double error(const int Ndim, const int Mdim, double * C, double * Cgold);


int main(int argc, char *argv[]) {

  using clock = std::chrono::high_resolution_clock;
  using timing = std::chrono::duration<double>; // seconds

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " Ndim Mdim Pdim" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Set input sizes
  const int Ndim = std::stol(argv[1]);
  const int Mdim = std::stol(argv[2]);
  const int Pdim = std::stol(argv[3]);
  if (Ndim < 1 || Mdim < 1 || Pdim < 1) {
    std::cerr << "Input error: matrix size cannot be zero/negative" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Print header
  std::cout
    << "Matrix multiplication benchmark" << std::endl << std::endl
    << "  A is " << Ndim << " by " << Pdim << std::endl
    << "  B is " << Pdim << " by " << Mdim << std::endl
    << "  C is " << Ndim << " by " << Mdim << std::endl
    << std::endl;

  // Allocate memory
  double *A = new double[Ndim*Pdim];
  double *B = new double[Pdim*Mdim];
  double *C = new double[Ndim*Mdim];
  double *Cgold = new double[Ndim*Mdim];

  init_input_matrices(Ndim, Mdim, Pdim, A, B);

  zero_matrix(Ndim, Mdim, C);

  auto tic = clock::now();
  matmul(Ndim, Mdim, Pdim, A, B, C);
  auto toc = clock::now();

  get_true_solution(Ndim, Mdim, Pdim, Cgold);

  double err = error(Ndim, Mdim, C, Cgold);

  if (err < 1.0E-8) {
    std::cout << "  Solution correct" << std::endl;
  } else {
    std::cout
      << "  Solution *NOT* correct" << std::endl
      << "    Error = " << err << std::endl;
  }

  // Print timings
  std::cout
    << "  matmul took " << timing{toc-tic}.count() << " s" << std::endl;

  // Deallocate memory
  delete[] A;
  delete[] B;
  delete[] C;
  delete[] Cgold;

}


// Matrix initalisation inspired by Tim Mattson's
// OpenMP tutorial codes.
// Input matrices generate a finite series so that
// their product is known.
//
// A: elements of rows run 1 to Pdim, scaled by Aval
// B: elements of cols run from 1 to Pdim, scaled by Bval
//    then, cols scaled by column number, 1 to Pdim
//
void init_input_matrices(const int Ndim, const int Mdim, const int Pdim, double *A, double *B) {

  // Initilise A
  for (int i = 0; i < Ndim; ++i) {
    for (int j = 0; j < Pdim; ++j) {
      A[i*Pdim + j] = Aval * static_cast<double>(j+1);
    }
  }

  // Initilise B
  for (int i = 0; i < Pdim; ++i) {
    for (int j = 0; j < Mdim; ++j) {
      B[i*Mdim + j] = static_cast<double>(j+1) * Bval * static_cast<double>(i+1);
    }
  }
}

void zero_matrix(const int Ndim, const int Mdim, double * C) {

  for (int i = 0; i < Ndim; ++i) {
    for (int j = 0; j < Mdim; ++j) {
      C[i*Mdim + j] = 0.0;
    }
  }
}


void matmul(const int Ndim, const int Mdim, const int Pdim, double *A, double *B, double *C) {

  for (int i = 0; i < Ndim; ++i) {
    for (int j = 0; j < Mdim; ++j) {
      for (int k = 0; k < Pdim; ++k) {
        C[i*Mdim + j] += A[i*Pdim + k] * B[k*Mdim + j];
      }
    }
  }

}

void get_true_solution(const int Ndim, const int Mdim, const int Pdim, double * C) {

  // Calculated from sum of k squared for k = 1 to P
  // Scale by AVAL and BVAL factors and column scaling of B
  double Ctmp = static_cast<double>(Pdim);
  double Cval = Ctmp * (Ctmp+1.0) * (2.0 * Ctmp + 1.0);
  Cval = Cval * Aval * Bval / 6.0;

  for (int i = 0; i < Ndim; ++i) {
    for (int j = 0; j < Mdim; ++j) {
      C[i*Mdim + j] = Cval * static_cast<double>(j+1);
    }
  }
}   

// Return the sum of the squares of the differences of the two input matrices
double error(const int Ndim, const int Mdim, double * C, double * Cgold) {

  double err = 0.0;

  for (int i = 0; i < Ndim; ++i) {
    for (int j = 0; j < Mdim; ++j) {
      double diff = C[i*Mdim + j] - Cgold[i*Mdim + j];
      err += diff * diff;
    }
  }

  return err;
}

