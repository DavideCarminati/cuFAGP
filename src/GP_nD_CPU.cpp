
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <cmath>
#include <random>
#include <assert.h>
#include <iostream>
#include <iostream>
#include <vector>
#include <chrono>
#include <boost/math/special_functions/hermite.hpp>
#include <chrono>
#include <fstream>
#ifdef _OPENMP
    #include <omp.h>
#endif

#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <Eigen/LU>

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

/**
 * Compute the eigenvectors and store them in the matrix Φ
*/
void eigenFunction( const MatrixXd &X, 
                    const MatrixXi &eigenvalue_combination, 
                    const double &epsilon, 
                    const double &alpha,
                    MatrixXd &Phi)
{
    const int N      = X.rows();
    const int n_comb = eigenvalue_combination.rows();
    const int n_dim  = eigenvalue_combination.cols();
    Phi.resize(N, n_comb);
    Phi.setOnes();
    // #pragma omp parallel for
    for (int ii = 0; ii < N; ii++)
    {
        for (int jj = 0; jj < n_comb; jj++)
        {
            double beta = pow( 1 + pow(2 * epsilon / alpha, 2), 0.25 );
            double delta = alpha * alpha / 2 * (beta * beta - 1);
            for (int p = 0; p < n_dim; p++)
            {
                double Gamma = sqrt( beta / (pow(2, eigenvalue_combination(jj, p) - 1) * tgamma((double)eigenvalue_combination(jj, p))) );
                Phi(ii, jj) *= Gamma * exp( -delta * pow(X(ii, p), 2) ) * \
                        boost::math::hermite(eigenvalue_combination(jj, p) - 1, alpha * beta * X(ii, p));
            }
        }
    }
}

/**
 * Compute eigenvalues and store them is a vector
*/
void eigenValues(   const MatrixXi &eigenvalue_combination, 
                    const double &epsilon, 
                    const double &alpha,
                    VectorXd &Lambda,
                    VectorXd &inv_Lambda)
{
    const int n_comb = eigenvalue_combination.rows();
    const int n_dim  = eigenvalue_combination.cols();
    Lambda.resize(n_comb);
    Lambda.setIdentity();
    inv_Lambda.resize(n_comb);
    // #pragma omp parallel for
    for (int ii = 0; ii < n_comb; ii++)
    {
        double beta = pow( 1 + pow(2 * epsilon / alpha, 2), 0.25 );
        double delta = alpha * alpha / 2 * (beta * beta - 1);
        for (int p = 0; p < n_dim; p++)
        {
            Lambda(ii) *= sqrt( alpha*alpha / (alpha*alpha + delta + epsilon*epsilon) ) \
                    * pow( epsilon*epsilon / (alpha*alpha + delta + epsilon*epsilon), eigenvalue_combination(ii, p) - 1 );
        }
        inv_Lambda(ii) = 1 / Lambda(ii);
    }
    return;
}

int main(int argc, char* argv[])
{
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

    const int N = y_train.rows();                       // Number of training points
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

    std::cout << "Executing on CPU...\n";
    auto cpu_start = std::chrono::steady_clock::now();

    // Compute eigenfunctions and eigenvalues
    MatrixXd K_app, Ks_app, Kss_app, Phi, Phip, Phi_T;
    VectorXd Lambda_vec, inv_Lambda_vec;
    SparseMatrix<double> Lambda(np, np);
    SparseMatrix<double> inv_Lambda(np, np);
    eigenFunction(x_train, eig_comb, epsilon, alpha, Phi);
    eigenFunction(x_test, eig_comb, epsilon, alpha, Phip);
    eigenValues(eig_comb, epsilon, alpha, Lambda_vec, inv_Lambda_vec);
    
    // Creating sparse diagonal matrices
    Lambda.reserve(VectorXi::Constant(np, 1));
    inv_Lambda.reserve(VectorXi::Constant(np, 1));
    for (int ii = 0; ii < np; ii++)
    {
        Lambda.insert(ii, ii) = Lambda_vec(ii);
        inv_Lambda.insert(ii, ii) = inv_Lambda_vec(ii);
    }
    Lambda.makeCompressed();
    inv_Lambda.makeCompressed();

    /**
     * USING CPU FOR COMPUTATION
    */

    MatrixXd Lambda_hat = inv_Lambda + sigma_n * Phi.transpose() * Phi;
    MatrixXd Kstar = Phip * Lambda * Phi.transpose();
    MatrixXd W = Kstar * (sigma_n * identity - sigma_n * sigma_n * Phi * Lambda_hat.inverse() * Phi.transpose());
    VectorXd y_star = W * y_train;
    
    auto elapsed_cpu = (std::chrono::steady_clock::now() - cpu_start);
    auto elapsed_time_cpu = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed_cpu).count();

    std::cout << "CPU took " << elapsed_time_cpu << "ms." << std::endl;
    std::time_t date_and_time = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());

    const static Eigen::IOFormat CSVFormat(Eigen::StreamPrecision, Eigen::DontAlignCols, ", ", "\n");

    // Saving logs to file
    std::ifstream read_logs(rel_log_path + "log_cpu.csv");
    bool logs_new = read_logs.peek() == std::ifstream::traits_type::eof();
    read_logs.close();

    std::ofstream logs(rel_log_path + "log_cpu.csv", std::ios_base::app);
    if (logs_new) logs << "elapsed_ms, N, N_test, dims, n, date_time\n";
    logs << elapsed_time_cpu << ", " << N << ", " << N_test << ", " << p << ", " << n << ", " << std::ctime(&date_and_time);

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