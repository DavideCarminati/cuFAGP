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
#include <vector>
#include <chrono>
#include <boost/math/special_functions/hermite.hpp>
#include <cublas_v2.h>
#include <chrono>
#include <fstream>
#include <cusparse.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/LU>
#include <cuLinearSolver.hpp>
#include "cuFAGPutils.hpp"
#include "cu_utils.hpp"

#define MAX_ERR 1e-6

using namespace Eigen;

bool verbose = false;

/**
 * Load .csv files and save the content into an Eigen3 matrix
*/
template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}

int main(int argc, char* argv[])
{
    // Display CUDA device information
    if (verbose)
    {
        if (DeviceInfo() > 0) return 1;
    }
    // Parse the input
    // Check the number of parameters
    if (argc < 4) {
        std::cerr << "Usage: " << argv[0] << " <Input folder path> <Output folder path> <Log file path>" << std::endl;
        return 1;
    }
    std::string rel_input_path(argv[1]);
    std::string rel_output_path(argv[2]);
    std::string rel_log_path(argv[3]);
    
    // Importing matrices from parsed folder
    MatrixXd x_train = load_csv<MatrixXd>(rel_input_path + "x_train.csv");
    MatrixXd x_test = load_csv<MatrixXd>(rel_input_path + "x_test.csv");
    MatrixXd y_train_tmp = load_csv<MatrixXd>(rel_input_path + "y_train.csv");
    MatrixXd y_test_tmp = load_csv<MatrixXd>(rel_input_path + "y_test.csv");
    MatrixXd eig_comb_in = load_csv<MatrixXd>(rel_input_path + "eigen_comb.csv");
    MatrixXi eig_comb = eig_comb_in.cast<int>();

    // Normalize
    VectorXd y_train = (y_train_tmp.array() - y_train_tmp.minCoeff()) / (y_train_tmp.maxCoeff() -  y_train_tmp.minCoeff());
    VectorXd y_test = (y_test_tmp.array() - y_test_tmp.minCoeff()) / (y_test_tmp.maxCoeff() - y_test_tmp.minCoeff());

    const int N = y_train.rows();                       // Number of train points
    const int N_test = y_test.rows();                   // Number of test points
    const int p = x_train.cols();                       // Number of dimensions of the problem
    const int n = eig_comb.maxCoeff();                  // Number of eigenvalues
    const int np = pow(n, p);                           // Number of total n^p combination of eigenvalues
    if (verbose)
    {
        std::cout << "\nDataset properties:\n";
        std::cout << "Number of training points: " << N << "\nNumber of test points: " << N_test << "\nNumber of problem dimension: " << p << 
                "\nNumber of eigenvalues: " << n << std::endl;
    }

    // GP parameters
    const double l = 1;                                 // Length scale
    const double epsilon = 1 / (sqrt(2) * l);           // Parameter ε
    const double alpha = 0.5;                           // Global length scale
    const double sigma_n = 1/pow(1e-3, 2);              // 1/σ² inverse of noise variance
    const double minus_sigma_n2 = -1 / pow(1e-3, 4);    // (1/σ²)² square of inverse of noise variance
    MatrixXd identity = MatrixXd::Identity(N, N);
    
    /**
     * USING GPU FOR COMPUTATION
    */

    std::cout << "Executing on GPU...\n";
    auto gpu_start = std::chrono::steady_clock::now();

    // Allocate device memory
    double *dev_x_train, *dev_x_test, *dev_y_train;                                         // Datasets
    // double *dev_l, *dev_epsilon, *dev_alpha, *dev_sigma_n, *dev_minus_sigma_n;              // Hyperparameters
    double *dev_Phi, *dev_Phip, *dev_Lambda, *dev_diag_Lambda, *dev_diag_inv_Lambda, 
                *dev_identity, *dev_Phi_T, *dev_lambdas, *dev_Lambda_hat;
    int *dev_eig_comb;
    CUDA_CHECK(cudaMalloc((void**)&dev_x_train, sizeof(double) * x_train.size()));
    CUDA_CHECK(cudaMalloc((void**)&dev_x_test, sizeof(double) * x_test.size()));

    CUDA_CHECK(cudaMalloc((void**)&dev_eig_comb, sizeof(int) * np * p));
    CUDA_CHECK(cudaMalloc((void**)&dev_Phi, sizeof(double) * N * pow(n, p)));
    CUDA_CHECK(cudaMalloc((void**)&dev_Phip, sizeof(double) * N_test * np));
    CUDA_CHECK(cudaMalloc((void**)&dev_diag_Lambda, sizeof(double) * np));
    CUDA_CHECK(cudaMalloc((void**)&dev_diag_inv_Lambda, sizeof(double) * np));

    // Results
    double *dev_y_star;
    CUDA_CHECK(cudaMalloc((void**)&dev_y_star, sizeof(double) * N_test));

    // Allocating temporary variables for the computation of W
    double *dev_tmp_W1, *dev_tmp_W2;
    // Allocating temporary variables for the computation of Phip * Lambda * Phi
    double *dev_tmp_Kstar, *dev_Kstar;
    double *dev_W;

    CUDA_CHECK(cudaMemcpy(dev_x_train, x_train.data(), sizeof(double) * x_train.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_x_test, x_test.data(), sizeof(double) * x_test.size(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dev_eig_comb, eig_comb.data(), sizeof(int) * eig_comb.size(), cudaMemcpyHostToDevice));

    cublasHandle_t handle;
    cublasCreate(&handle);

    cusparseHandle_t handleSp;
    CUSPARSE_CHECK(cusparseCreate(&handleSp));

    const double one = 1.0, zero = 0.0;
    const int incx = 1;
    const int incy = 1;

    dim3 block_size(16, 64);
    dim3 grid_size;
    grid_size.x = (np + block_size.x - 1) / block_size.x;
    grid_size.y = (N + block_size.y - 1) / block_size.y;
    if (verbose)
    {
        std::cout << "Threads per block: " << block_size.x * block_size.y << ". Blocks per grid: " << grid_size.x * grid_size.y << std::endl;
        std::cout << "Biggest matrix processed in one GPU cycle: " << block_size.y * grid_size.y << "x" << block_size.x * grid_size.x << std::endl;
    }
    double beta = pow( 1 + pow(2 * epsilon / alpha, 2), 0.25 );
    double delta = alpha * alpha / 2 * (beta * beta - 1);

    // Compute eigenvectors and eigenvalue of the square exponential kernel decomposition
    eigenFunction<<<grid_size, block_size>>>(dev_x_train, N, p, dev_eig_comb, np, epsilon, alpha, beta, delta, dev_Phi);
    eigenFunction<<<grid_size, block_size>>>(dev_x_test, N_test, p, dev_eig_comb, np, epsilon, alpha, beta, delta, dev_Phip);
    eigenValues<<<grid_size, block_size>>>(dev_eig_comb, np, p, epsilon, alpha, beta, delta, dev_diag_Lambda, dev_diag_inv_Lambda);
    CUDA_CHECK(cudaFree(dev_x_train));
    CUDA_CHECK(cudaFree(dev_x_test));
    CUDA_CHECK(cudaFree(dev_eig_comb));

    // Computing Lambda_hat
    CUDA_CHECK(cudaMalloc((void**)&dev_Lambda_hat, sizeof(double) * np * np));
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N, np, np, N,
                    &sigma_n, dev_Phi, N, dev_Phi, N, &zero, dev_Lambda_hat, np));

    const int inc_diag_mat = np + 1;
    CUBLAS_CHECK(cublasDaxpy(handle, np, &one, dev_diag_inv_Lambda, incx, dev_Lambda_hat, inc_diag_mat));
    CUDA_CHECK(cudaFree(dev_diag_inv_Lambda));

    // Computing Lambda_hat^-1 * Phi.transpose(). Solution is stored in dev_Phi_T
    CUDA_CHECK(cudaMalloc((void**)&dev_Phi_T, sizeof(double) * N * np));
    CUBLAS_CHECK(cublasDgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, np, N, &one, dev_Phi, N, &zero, nullptr, np, dev_Phi_T, np));
    cuLinearSolver(dev_Lambda_hat, np, dev_Phi_T, N);
    CUDA_CHECK(cudaFree(dev_Lambda_hat));

    // Computing Sigma_n^-1 - sigma_n^2 * Phi * (Lambda_hat^-1 * Phi.transpose()). Lambda_hat^-1 * Phi.transpose() was computed using
    // cuLinearSolver(). Solution is stored in dev_tmp_W2 (initialized as identity matrix to pass Sigma_n^-1)
    CUDA_CHECK(cudaMalloc((void**)&dev_tmp_W2, sizeof(double) * N * N));
    CUDA_CHECK(cudaMemcpy(dev_tmp_W2, identity.data(), sizeof(double) * N * N, cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N, N, np, &minus_sigma_n2, dev_Phi, N, dev_Phi_T, np, &sigma_n, dev_tmp_W2, N));
    CUDA_CHECK(cudaFree(dev_Phi_T));
    
    // This is a dense-sparse matrix multiplication
    CUDA_CHECK(cudaMalloc((void**)&dev_tmp_Kstar, sizeof(double) * N * np));
    cusparseSpMatDescr_t Lambda;
    cusparseDnMatDescr_t Phi, tmp_Kstar;
    void*                dBuffer = NULL;
    size_t               bufferSize = 0;
    VectorXi Lambda_elem_indx = VectorXi::LinSpaced(np, 0, np - 1);
    int *dev_Lambda_elem_indx;
    CUDA_CHECK(cudaMalloc((void**)&dev_Lambda_elem_indx, sizeof(int) * np));
    CUDA_CHECK(cudaMemcpy(dev_Lambda_elem_indx, Lambda_elem_indx.data(), sizeof(int) * np, cudaMemcpyHostToDevice));
    CUSPARSE_CHECK(cusparseCreateCoo(   &Lambda, np, np, np, dev_Lambda_elem_indx, dev_Lambda_elem_indx, dev_diag_Lambda, 
                                        CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));
    CUSPARSE_CHECK(cusparseCreateDnMat(&Phi, N, np, N, dev_Phi, CUDA_R_64F, CUSPARSE_ORDER_COL));
    CUSPARSE_CHECK(cusparseCreateDnMat(&tmp_Kstar, np, N, np, dev_tmp_Kstar, CUDA_R_64F, CUSPARSE_ORDER_COL)); 
    CUSPARSE_CHECK(cusparseSpMM_bufferSize(handleSp, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                        &one, Lambda, Phi, &zero, tmp_Kstar, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    CUDA_CHECK(cudaMalloc(&dBuffer, bufferSize));
    CUSPARSE_CHECK(cusparseSpMM(handleSp, CUSPARSE_OPERATION_NON_TRANSPOSE, CUSPARSE_OPERATION_TRANSPOSE,
                                        &one, Lambda, Phi, &zero, tmp_Kstar, CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
    CUSPARSE_CHECK(cusparseDestroySpMat(Lambda));
    CUSPARSE_CHECK(cusparseDestroyDnMat(Phi));
    CUSPARSE_CHECK(cusparseDestroyDnMat(tmp_Kstar));
    CUSPARSE_CHECK(cusparseDestroy(handleSp));

    CUDA_CHECK(cudaFree(dev_Phi));
    CUDA_CHECK(cudaFree(dev_diag_Lambda));

    CUDA_CHECK(cudaMalloc((void**)&dev_Kstar, sizeof(double) * N_test * N));
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_test, N, np, &one, dev_Phip, N_test, dev_tmp_Kstar, np, &zero, dev_Kstar, N_test));
    
    CUDA_CHECK(cudaFree(dev_Phip));
    CUDA_CHECK(cudaFree(dev_tmp_Kstar));

    // Computing W
    CUDA_CHECK(cudaMalloc((void**)&dev_W, sizeof(double) * N_test * N));
    CUBLAS_CHECK(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, N_test, N, N, &one, dev_Kstar, N_test, dev_tmp_W2, N, &zero, dev_W, N_test));
    CUDA_CHECK(cudaFree(dev_tmp_W2));
    CUDA_CHECK(cudaFree(dev_Kstar));

    // Computing predictive posterior mean y_star and covariance cov_star
    CUDA_CHECK(cudaMalloc((void**)&dev_y_train, sizeof(double) * N));
    CUDA_CHECK(cudaMemcpy(dev_y_train, y_train.data(), sizeof(double) * y_train.size(), cudaMemcpyHostToDevice));
    CUBLAS_CHECK(cublasDgemv(handle, CUBLAS_OP_N, N_test, N, &one, dev_W, N_test, dev_y_train, incx, &zero, dev_y_star, incy));

    VectorXd y_star(N_test);
    CUDA_CHECK(cudaMemcpy(y_star.data(), dev_y_star, sizeof(double) * N_test, cudaMemcpyDeviceToHost));

    auto elapsed_gpu = (std::chrono::steady_clock::now() - gpu_start);
    auto elapsed_time_gpu = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_gpu).count();
    
    std::cout << "GPU took " << elapsed_time_gpu << "ms." << std::endl;
    std::time_t date_and_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    // Saving results in output folder
    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    // Saving logs to file
    std::ifstream read_logs(rel_log_path + "log_gpu.csv");
    bool logs_new = read_logs.peek() == std::ifstream::traits_type::eof();
    read_logs.close();

    std::ofstream logs(rel_log_path + "log_gpu.csv", std::ios_base::app);
    if (logs_new) logs << "elapsed_ms, N, N_test, dims, n, date_time\n";
    logs << elapsed_time_gpu << ", " << N << ", " << N_test << ", " << p << ", " << n << ", " << std::ctime(&date_and_time);// << std::endl;

    std::ofstream file_x_train(rel_output_path + "x_train.csv");
    file_x_train << x_train.format(CSVFormat);

    std::ofstream file_y_train(rel_output_path + "y_train.csv");
    file_y_train << y_train.format(CSVFormat);

    std::ofstream file_x_test(rel_output_path + "x_test.csv");
    file_x_test << x_test.format(CSVFormat);

    std::ofstream file_y_test(rel_output_path + "y_test.csv");
    file_y_test << y_test.format(CSVFormat);

    std::ofstream file_y_star(rel_output_path + "y_predicted.csv");
    file_y_star << y_star.format(CSVFormat);

}