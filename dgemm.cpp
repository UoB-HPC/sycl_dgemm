// Copyright (c) 2021 SYCL DGEMM Benchmark authors
// SPDX-License-Identifier: MIT

#include <iostream>
#include <chrono>

#include <CL/sycl.hpp>


// Matrix multiplication benchmark
//
//  Computes C = A x B for matrices A, B and C of sizes:
//  A is N x P
//  B is P x M
//  C is N x M
//

const double Aval = 0.30;
const double Bval = 0.50;

void init_input_matrices(cl::sycl::queue& Q, const size_t Ndim, const size_t Mdim, const size_t Pdim, cl::sycl::buffer<double,2>& A, cl::sycl::buffer<double,2>& B);
void zero_matrix(cl::sycl::queue& Q, const size_t Ndim, const size_t Mdim, cl::sycl::buffer<double,2>& C);
void matmul(cl::sycl::queue& Q, const size_t Ndim, const size_t Mdim, const size_t Pdim, cl::sycl::buffer<double,2>& A, cl::sycl::buffer<double,2>& B, cl::sycl::buffer<double,2>& C);
void matmul_blocked(cl::sycl::queue& Q, const size_t Ndim, const size_t Mdim, const size_t Pdim, cl::sycl::buffer<double,2>& A, cl::sycl::buffer<double,2>& B, cl::sycl::buffer<double,2>& C);
void matmul_hipar(cl::sycl::queue& Q, const size_t Ndim, const size_t Mdim, const size_t Pdim, cl::sycl::buffer<double,2>& A, cl::sycl::buffer<double,2>& B, cl::sycl::buffer<double,2>& C);
void get_true_solution(const int Ndim, const int Mdim, const int Pdim, double * C);
double error(const int Ndim, const int Mdim, double * C, double * Cgold);


#ifdef __HIPSYCL__
void matmul_scopedpar(cl::sycl::queue& Q, const size_t Ndim, const size_t Mdim, const size_t Pdim, cl::sycl::buffer<double,2>& A, cl::sycl::buffer<double,2>& B, cl::sycl::buffer<double,2>& C);
#endif

int main(int argc, char *argv[]) {

  using clock = std::chrono::high_resolution_clock;
  using timing = std::chrono::duration<double>; // seconds

  if (argc != 4) {
    std::cerr << "Usage: " << argv[0] << " Ndim Mdim Pdim" << std::endl;
    exit(EXIT_FAILURE);
  }

  // Set input sizes
  const size_t Ndim = std::stol(argv[1]);
  const size_t Mdim = std::stol(argv[2]);
  const size_t Pdim = std::stol(argv[3]);
  if (Ndim < 1 || Mdim < 1 || Pdim < 1) {
    std::cerr << "Input error: matrix size cannot be zero/negative" << std::endl;
    exit(EXIT_FAILURE);
  }

  cl::sycl::queue Q;

  // Print header
  std::cout
    << "Matrix multiplication benchmark" << std::endl << std::endl
    << "  A is " << Ndim << " by " << Pdim << std::endl
    << "  B is " << Pdim << " by " << Mdim << std::endl
    << "  C is " << Ndim << " by " << Mdim << std::endl
    << std::endl
    << "  Using SYCL device: " << Q.get_device().get_info<cl::sycl::info::device::name>() << std::endl
    << std::endl;

  // Allocate memory
  cl::sycl::buffer<double, 2> A({Ndim,Pdim});
  cl::sycl::buffer<double, 2> B({Pdim,Mdim});
  cl::sycl::buffer<double, 2> C({Ndim,Mdim});

  double *Cgold = new double[Ndim*Mdim];
  get_true_solution(Ndim, Mdim, Pdim, Cgold);

  init_input_matrices(Q, Ndim, Mdim, Pdim, A, B);


  //
  // Initial version
  //
  {
    std::cout << "  Simple version:" << std::endl << std::endl;

    // Make sure previous work finished for accurate timing
    zero_matrix(Q, Ndim, Mdim, C);
    Q.wait_and_throw();

    auto tic = clock::now();
    matmul(Q, Ndim, Mdim, Pdim, A, B, C);
    auto toc = clock::now();


    double err = error(Ndim, Mdim, C.get_access<cl::sycl::access_mode::read>().get_pointer(), Cgold);

    if (err < 1.0E-8) {
      std::cout << "  Solution correct" << std::endl;
    } else {
      std::cout
        << "  Solution *NOT* correct" << std::endl
        << "    Error = " << err << std::endl;
    }

    // Print timings
    std::cout
      << "  matmul took " << timing{toc-tic}.count() << " s" << std::endl
      << "  GFLOP/s: " << 1.0E-9 * 2.0 * Ndim * Mdim * Pdim / (timing{toc-tic}.count()) << std::endl;

    std::cout << "  --------------------------------" << std::endl << std::endl;
  }

  //
  // nd_range blocked version
  //
  {
    std::cout << "  Blocked nd_range version:" << std::endl << std::endl;

    // Make sure previous work finished for accurate timing
    zero_matrix(Q, Ndim, Mdim, C);
    Q.wait_and_throw();

    auto tic = clock::now();
    matmul_blocked(Q, Ndim, Mdim, Pdim, A, B, C);
    auto toc = clock::now();


    double err = error(Ndim, Mdim, C.get_access<cl::sycl::access_mode::read>().get_pointer(), Cgold);

    if (err < 1.0E-8) {
      std::cout << "  Solution correct" << std::endl;
    } else {
      std::cout
        << "  Solution *NOT* correct" << std::endl
        << "    Error = " << err << std::endl;
    }

    // Print timings
    std::cout
      << "  matmul took " << timing{toc-tic}.count() << " s" << std::endl
      << "  GFLOP/s: " << 1.0E-9 * 2.0 * Ndim * Mdim * Pdim / (timing{toc-tic}.count()) << std::endl;

    std::cout << "  --------------------------------" << std::endl << std::endl;
  }

  //
  // Hierarchical parallel version
  //
  {
    std::cout << "  Hierarchical parallel version:" << std::endl << std::endl;

    zero_matrix(Q, Ndim, Mdim, C);
    Q.wait_and_throw();

    auto tic = clock::now();
    matmul_hipar(Q, Ndim, Mdim, Pdim, A, B, C);
    auto toc = clock::now();

    double err = error(Ndim, Mdim, C.get_access<cl::sycl::access_mode::read>().get_pointer(), Cgold);

    if (err < 1.0E-8) {
      std::cout << "  Solution correct" << std::endl;
    } else {
      std::cout
        << "  Solution *NOT* correct" << std::endl
        << "    Error = " << err << std::endl;
    }

    // Print timings
    std::cout
      << "  matmul took " << timing{toc-tic}.count() << " s" << std::endl
      << "  GFLOP/s: " << 1.0E-9 * 2.0 * Ndim * Mdim * Pdim / (timing{toc-tic}.count()) << std::endl;

    std::cout << "  --------------------------------" << std::endl << std::endl;
  }

  //
  // hipSYCL Scoped Parallelism extension
  //
  #ifdef __HIPSYCL__
  {
    std::cout << "  Scoped parallelism version:" << std::endl << std::endl;

    zero_matrix(Q, Ndim, Mdim, C);
    Q.wait_and_throw();

    auto tic = clock::now();
    matmul_scopedpar(Q, Ndim, Mdim, Pdim, A, B, C);
    auto toc = clock::now();

    double err = error(Ndim, Mdim, C.get_access<cl::sycl::access_mode::read>().get_pointer(), Cgold);

    if (err < 1.0E-8) {
      std::cout << "  Solution correct" << std::endl;
    } else {
      std::cout
        << "  Solution *NOT* correct" << std::endl
        << "    Error = " << err << std::endl;
    }

    // Print timings
    std::cout
      << "  matmul took " << timing{toc-tic}.count() << " s" << std::endl
      << "  GFLOP/s: " << 1.0E-9 * 2.0 * Ndim * Mdim * Pdim / (timing{toc-tic}.count()) << std::endl;

    std::cout << "  --------------------------------" << std::endl << std::endl;
  }
  #endif


  // Deallocate memory
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
void init_input_matrices(cl::sycl::queue& Q, const size_t Ndim, const size_t Mdim, const size_t Pdim, cl::sycl::buffer<double,2>& A, cl::sycl::buffer<double,2>& B) {

  // Initilise A
  Q.submit([&](cl::sycl::handler &cgh) {
    //cl::sycl::accessor a {A, cgh, cl::sycl::write_only, cl::sycl::noinit};
    auto a = A.get_access<cl::sycl::access_mode::write>(cgh);

    cgh.parallel_for(cl::sycl::range<2>{Ndim, Pdim}, [=](cl::sycl::id<2> idx) {
      a[idx] = Aval * static_cast<double>(idx[1] + 1);
    });
  });

  // Initilise B
  Q.submit([&](cl::sycl::handler &cgh) {
    //cl::sycl::accessor b {B, cgh, cl::sycl::write_only, cl::sycl::noinit};
    auto b = B.get_access<cl::sycl::access_mode::write>(cgh);

    cgh.parallel_for(cl::sycl::range<2>{Pdim, Mdim}, [=](cl::sycl::id<2> idx) {
      b[idx] = static_cast<double>(idx[1] + 1) * Bval * static_cast<double>(idx[0] + 1);
    });
  });
}

void zero_matrix(cl::sycl::queue& Q, const size_t Ndim, const size_t Mdim, cl::sycl::buffer<double,2>& C) {

  Q.submit([&](cl::sycl::handler &cgh) {
    //cl::sycl::accessor c {C, cgh, cl::sycl::write_only, cl::sycl::noinit};
    auto c = C.get_access<cl::sycl::access_mode::write>(cgh);

    cgh.parallel_for(cl::sycl::range<2>{Ndim, Mdim}, [=](cl::sycl::id<2> idx) {
      c[idx] = 0.0;
    });
  });
}


void matmul(cl::sycl::queue& Q, const size_t Ndim, const size_t Mdim, const size_t Pdim, cl::sycl::buffer<double,2>& A, cl::sycl::buffer<double,2>& B, cl::sycl::buffer<double,2>& C) {

  Q.submit([&](cl::sycl::handler &cgh) {
    //cl::sycl::accessor a {A, cgh, cl::sycl::read_only};
    //cl::sycl::accessor b {B, cgh, cl::sycl::read_only};
    //cl::sycl::accessor c {C, cgh, cl::sycl::read_write};
    auto a = A.get_access<cl::sycl::access_mode::read>(cgh);
    auto b = B.get_access<cl::sycl::access_mode::read>(cgh);
    auto c = C.get_access<cl::sycl::access_mode::read_write>(cgh);

    cgh.parallel_for(cl::sycl::range<2>{Ndim, Mdim}, [=](cl::sycl::id<2> idx) {
      const size_t i = idx[0];
      const size_t j = idx[1];
      for (int k = 0; k < Pdim; ++k) {
        c[idx] += a[i][k] * b[k][j];
      }
    });
  }).wait();

}

void matmul_blocked(cl::sycl::queue& Q, const size_t Ndim, const size_t Mdim, const size_t Pdim, cl::sycl::buffer<double,2>& A, cl::sycl::buffer<double,2>& B, cl::sycl::buffer<double,2>& C) {

  const size_t Bsize = 16;
  assert(Ndim % Bsize == 0);
  assert(Mdim % Bsize == 0);
  assert(Pdim % Bsize == 0);

  Q.submit([&](cl::sycl::handler &cgh) {
    //cl::sycl::accessor a {A, cgh, cl::sycl::read_only};
    //cl::sycl::accessor b {B, cgh, cl::sycl::read_only};
    //cl::sycl::accessor c {C, cgh, cl::sycl::read_write};
    auto a = A.get_access<cl::sycl::access_mode::read>(cgh);
    auto b = B.get_access<cl::sycl::access_mode::read>(cgh);
    auto c = C.get_access<cl::sycl::access_mode::read_write>(cgh);

    cl::sycl::accessor<double, 2, cl::sycl::access_mode::read_write, cl::sycl::access::target::local> Awrk({Bsize, Bsize}, cgh);
    cl::sycl::accessor<double, 2, cl::sycl::access_mode::read_write, cl::sycl::access::target::local> Bwrk({Bsize, Bsize}, cgh);

    cgh.parallel_for(cl::sycl::nd_range<2>{{Ndim, Mdim}, {Bsize, Bsize}}, [=](cl::sycl::nd_item<2> idx) {

      // This work-item will compute C(i,j)
      const size_t i = idx.get_global_id(0);
      const size_t j = idx.get_global_id(1);

      // Element C(i,j) is in block C(Iblk, Jblk)
      const size_t Iblk = idx.get_group(0);
      const size_t Jblk = idx.get_group(1);

      // C(i,j) is element C(iloc, jloc) of block C(Iblk, Jblk)
      const size_t iloc = idx.get_local_id(0);
      const size_t jloc = idx.get_local_id(1);

      // Number of blocks
      const size_t Nblk = Ndim / Bsize;
      const size_t Mblk = Mdim / Bsize;
      const size_t Pblk = Pdim / Bsize;

      for (int Kblk = 0; Kblk < Pblk; ++Kblk) {

        // Copy A and B into local memory
        Awrk[iloc][jloc] = a[Iblk*Bsize+iloc][Kblk*Bsize+jloc];
        Bwrk[iloc][jloc] = b[Kblk*Bsize+iloc][Jblk*Bsize+jloc];
        //cl::sycl::group_barrier(idx.get_group());
        idx.barrier();

        // Compute matmul for block
        for (int kloc = 0; kloc < Bsize; ++kloc) {
          c[i][j] += Awrk[iloc][kloc] * Bwrk[kloc][jloc];
        }
        //cl::sycl::group_barrier(idx.get_group());
        idx.barrier();
      }
    });
  }).wait();

}

void matmul_hipar(cl::sycl::queue& Q, const size_t Ndim, const size_t Mdim, const size_t Pdim, cl::sycl::buffer<double,2>& A, cl::sycl::buffer<double,2>& B, cl::sycl::buffer<double,2>& C) {

  const size_t Bsize = 16;
  assert(Ndim % Bsize == 0);
  assert(Mdim % Bsize == 0);
  assert(Pdim % Bsize == 0);

  // Number of blocks
  const size_t Nblk = Ndim / Bsize;
  const size_t Mblk = Mdim / Bsize;
  const size_t Pblk = Pdim / Bsize;


  Q.submit([&](cl::sycl::handler &cgh) {
    //cl::sycl::accessor a {A, cgh, cl::sycl::read_only};
    //cl::sycl::accessor b {B, cgh, cl::sycl::read_only};
    //cl::sycl::accessor c {C, cgh, cl::sycl::read_write};
    auto a = A.get_access<cl::sycl::access_mode::read>(cgh);
    auto b = B.get_access<cl::sycl::access_mode::read>(cgh);
    auto c = C.get_access<cl::sycl::access_mode::read_write>(cgh);

    cgh.parallel_for_work_group(cl::sycl::range<2>{Nblk, Mblk}, cl::sycl::range<2>{Bsize, Bsize}, [=](cl::sycl::group<2> g) {

      double Awrk[Bsize][Bsize];
      double Bwrk[Bsize][Bsize];

      // Element C(i,j) is in block C(Iblk, Jblk)
      const size_t Iblk = g[0];
      const size_t Jblk = g[1];

      for (int Kblk = 0; Kblk < Pblk; ++Kblk) {

        // Copy A and B into local memory
        g.parallel_for_work_item([&](cl::sycl::h_item<2> idx) {
          const size_t iloc = idx.get_local_id(0);
          const size_t jloc = idx.get_local_id(1);
          Awrk[iloc][jloc] = a[Iblk*Bsize+iloc][Kblk*Bsize+jloc];
          Bwrk[iloc][jloc] = b[Kblk*Bsize+iloc][Jblk*Bsize+jloc];
        });

        // Compute matmul for block
        g.parallel_for_work_item([&](cl::sycl::h_item<2> idx) {
          const size_t iloc = idx.get_local_id(0);
          const size_t jloc = idx.get_local_id(1);
          for (int kloc = 0; kloc < Bsize; ++kloc) {
            c[idx.get_global_id()] += Awrk[iloc][kloc] * Bwrk[kloc][jloc];
          }
        });
      }
    });
  }).wait();

}

#ifdef __HIPSYCL__
void matmul_scopedpar(cl::sycl::queue& Q, const size_t Ndim, const size_t Mdim, const size_t Pdim, cl::sycl::buffer<double,2>& A, cl::sycl::buffer<double,2>& B, cl::sycl::buffer<double,2>& C) {

  const size_t Bsize = 16;
  assert(Ndim % Bsize == 0);
  assert(Mdim % Bsize == 0);
  assert(Pdim % Bsize == 0);

  // Number of blocks
  const size_t Nblk = Ndim / Bsize;
  const size_t Mblk = Mdim / Bsize;
  const size_t Pblk = Pdim / Bsize;


  Q.submit([&](cl::sycl::handler &cgh) {
    //cl::sycl::accessor a {A, cgh, cl::sycl::read_only};
    //cl::sycl::accessor b {B, cgh, cl::sycl::read_only};
    //cl::sycl::accessor c {C, cgh, cl::sycl::read_write};
    auto a = A.get_access<cl::sycl::access_mode::read>(cgh);
    auto b = B.get_access<cl::sycl::access_mode::read>(cgh);
    auto c = C.get_access<cl::sycl::access_mode::read_write>(cgh);

    cgh.parallel(cl::sycl::range<2>{Nblk, Mblk}, cl::sycl::range<2>{Bsize, Bsize}, [=](auto g) {

      cl::sycl::local_memory<double[Bsize][Bsize], decltype(g)> Awrk;
      cl::sycl::local_memory<double[Bsize][Bsize], decltype(g)> Bwrk;

      // Element C(i,j) is in block C(Iblk, Jblk)
      const size_t Iblk = g[0];
      const size_t Jblk = g[1];

      for (int Kblk = 0; Kblk < Pblk; ++Kblk) {

        // Copy A and B into local memory
        cl::sycl::distribute_items_and_wait(g, [&](cl::sycl::s_item<2> idx) {
          const size_t iloc = idx.get_innermost_local_id(0);
          const size_t jloc = idx.get_innermost_local_id(1);
          Awrk[iloc][jloc] = a[Iblk*Bsize+iloc][Kblk*Bsize+jloc];
          Bwrk[iloc][jloc] = b[Kblk*Bsize+iloc][Jblk*Bsize+jloc];
        });

        // Compute matmul for block
        cl::sycl::distribute_items(g, [&](cl::sycl::s_item<2> idx) {
          const size_t iloc = idx.get_innermost_local_id(0);
          const size_t jloc = idx.get_innermost_local_id(1);
          for (int kloc = 0; kloc < Bsize; ++kloc) {
            c[idx.get_global_id()] += Awrk[iloc][kloc] * Bwrk[kloc][jloc];
          }
        });
      }
    });
  }).wait();

}
#endif

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

