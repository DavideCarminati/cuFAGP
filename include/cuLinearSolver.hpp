#pragma once

#include <cstdio>
#include <cstdlib>
#include <vector>
#include <Eigen/Dense>

#include <cuda_runtime.h>
#include <cusolverDn.h>

#include "cu_utils.hpp"

int cuLinearSolver( double *d_A, // Device address of matrix A 
                    const int64_t n,   // rows of A (square matrix)
                    double *d_B, // Device address of matrix B and output of solution
                    const int64_t nrhs // cols of B
                    );