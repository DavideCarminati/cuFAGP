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
                    const int N,                // Number of samples in dataset
                    const int n_dim,            // Number of dimensions of the problem
                    const int *eigenvalue_comb, 
                    const int n_comb,           // length of eigenvalue_comb
                    const double epsilon, 
                    const double alpha, 
                    const double beta,
                    const double delta,
                    double *Phi_out)
{
    int tid_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tid_y = blockIdx.y * blockDim.y + threadIdx.y;

    if (tid_x < n_comb && tid_y < N)
    {
        // Fill the Phi matrix in a parallel fashion
        Phi_out[tid_y + N * tid_x] = 1.0;
        
        for (int p = 0; p < n_dim; p++)
        {
            int idx_x = tid_y + p * N;
            int idx_eig_comb = tid_x + p * n_comb;
            double Gamma = sqrt( beta / (pow(2, eigenvalue_comb[idx_eig_comb] - 1) * tgamma((double)eigenvalue_comb[idx_eig_comb])) );
            Phi_out[tid_y + N * tid_x] *= Gamma * exp( -delta * pow(x[idx_x], 2) ) * 
                    cuHermite(eigenvalue_comb[idx_eig_comb] - 1, alpha * beta * x[idx_x]);
        }
    }
    __syncthreads();
    return;
}

__global__
void eigenValues(const int *eigenvalue_comb, 
                        const int n_comb, 
                        const int n_dim, 
                        const double epsilon, 
                        const double alpha,
                        const double beta,
                        const double delta, 
                        double *Lambda, 
                        double *inv_Lambda)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid < n_comb)
    {
        Lambda[tid] = 1.0;
        for (int p = 0; p < n_dim; p++)
        {
            Lambda[tid] *= sqrt( alpha*alpha / (alpha*alpha + delta + epsilon*epsilon) ) \
                    * pow( epsilon*epsilon / (alpha*alpha + delta + epsilon*epsilon), eigenvalue_comb[tid + p * n_comb] - 1 );
        }
        inv_Lambda[tid] = 1 / Lambda[tid];
        
    }
    __syncthreads();
    return;
}