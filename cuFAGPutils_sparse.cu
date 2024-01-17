#include "cuFAGPutils.hpp"

using namespace Eigen;

__device__
double cuHermite(const int n, const double x)   // recursive version
{
    if (n <= 0)      
        return -1;
    else if (n == 1) 
        return 1;
    else if (n == 2) 
        return 2 * x;
    else 
        return 2 * x * cuHermite(n-1, x) - 2 * n * cuHermite(n - 2, x);
}

__global__
void eigenFunction( double *x, 
                    const int N,                // # of samples in dataset
                    const int n_dim,            // # of dimensions of the problem
                    const int *eigenvalue_comb, 
                    const int n_comb,           // length of eigenvalue_comb
                    const double epsilon, 
                    const double alpha, 
                    double *Phi_out,
                    double *Phi_T_out)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;

    if (tid < N * n_comb)
    {
        // Fill the Phi matrix in a parallel fashion
        Phi_out[tid] = 1.0;
        double beta = pow( 1 + pow(2 * epsilon / alpha, 2), 0.25 );
        double delta = alpha * alpha / 2 * (beta * beta - 1);
        for (int p = 0; p < n_dim; p++)
        {
            int idx_x = tid % N + p * N;
            int idx_eig_comb = static_cast<int>(floor(tid / (double)N)) + p * n_comb;
            double Gamma = sqrt( beta / (pow(2, eigenvalue_comb[idx_eig_comb] - 1) * tgamma((double)eigenvalue_comb[idx_eig_comb])) );
            Phi_out[tid] *= Gamma * exp( -delta * pow(x[idx_x], 2) ) * cuHermite(eigenvalue_comb[idx_eig_comb] - 1, alpha * beta * x[idx_x]);
        }
    }
    __syncthreads();
    if (Phi_T_out)
    {
        Map<MatrixXd> Phi_T(Phi_T_out, n_comb, N);
        Map<MatrixXd> Phi(Phi_out, N, n_comb);
        Phi_T = Phi.transpose();
    }
    return;
}

__global__
void eigenValues(   const int *eigenvalue_comb, 
                    const int n_comb, 
                    const int n_dim, 
                    const double epsilon, 
                    const double alpha, 
                    double *Lambda, 
                    double *inv_Lambda)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    Map<MatrixXd> Lambda_eigen(Lambda, n_comb, n_comb);
    Lambda_eigen.setIdentity();
    Map<MatrixXd> inv_Lambda_eigen(inv_Lambda, n_comb, n_comb);
    if (tid < n_comb)
    {
        double beta = pow( 1 + pow(2 * epsilon / alpha, 2), 0.25 );
        double delta = alpha * alpha / 2 * (beta * beta - 1);
        for (int p = 0; p < n_dim; p++)
        {
            Lambda_eigen(tid,tid) *= sqrt( alpha*alpha / (alpha*alpha + delta + epsilon*epsilon) ) \
                    * pow( epsilon*epsilon / (alpha*alpha + delta + epsilon*epsilon), eigenvalue_comb[tid + p * n_comb] - 1 );
        }
        inv_Lambda_eigen(tid,tid) = 1 / Lambda_eigen(tid,tid);
        
    }
    __syncthreads();
    return;
}