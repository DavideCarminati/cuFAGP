#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <random>
#include <assert.h>
#include <iostream>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cub/cub.cuh>
#include <iostream>
#include <cublas_v2.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <cuLinearSolver.hpp>

__device__
double cuHermite(const int n, const double x);

// __global__
// void eigenFunction( double *x, 
//                     const int N,                // # of samples in dataset
//                     const int n_dim,            // # of dimensions of the problem
//                     const int *eigenvalue_comb, 
//                     const int n_comb,           // length of eigenvalue_comb
//                     const double epsilon, 
//                     const double alpha, 
//                     double *Phi_out,
//                     double *Phi_T_out);

__global__
void eigenFunction( double *x, 
                    const int N,                // # of samples in dataset
                    const int n_dim,            // # of dimensions of the problem
                    const int *eigenvalue_comb, 
                    const int n_comb,           // length of eigenvalue_comb
                    const double epsilon, 
                    const double alpha, 
                    const double beta,
                    const double delta,
                    double *Phi_out,
                    double *Phi_T_out);

__global__
void eigenValues(   const int *eigenvalue_comb, 
                    const int n_comb, 
                    const int n_dim, 
                    const double epsilon, 
                    const double alpha, 
                    double *Lambda, 
                    double *inv_Lambda);

__global__
void eigenValues_sparse(const int *eigenvalue_comb, 
                        const int n_comb, 
                        const int n_dim, 
                        const double epsilon, 
                        const double alpha, 
                        const double beta,
                        const double delta,
                        double *Lambda, 
                        double *inv_Lambda);