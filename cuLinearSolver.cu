/*
 * Copyright 2020 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO LICENSEE:
 *
 * This source code and/or documentation ("Licensed Deliverables") are
 * subject to NVIDIA intellectual property rights under U.S. and
 * international Copyright laws.
 *
 * These Licensed Deliverables contained herein is PROPRIETARY and
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and
 * conditions of a form of NVIDIA software license agreement by and
 * between NVIDIA and Licensee ("License Agreement") or electronically
 * accepted by Licensee.  Notwithstanding any terms or conditions to
 * the contrary in the License Agreement, reproduction or disclosure
 * of the Licensed Deliverables to any third party without the express
 * written consent of NVIDIA is prohibited.
 *
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
 * SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
 * PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
 * NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
 * DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
 * NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
 * NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
 * LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
 * SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
 * DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
 * WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
 * ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
 * OF THESE LICENSED DELIVERABLES.
 *
 * U.S. Government End Users.  These Licensed Deliverables are a
 * "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
 * 1995), consisting of "commercial computer software" and "commercial
 * computer software documentation" as such terms are used in 48
 * C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
 * only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
 * 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
 * U.S. Government End Users acquire the Licensed Deliverables with
 * only those rights set forth herein.
 *
 * Any use of the Licensed Deliverables in individual and commercial
 * software must include, in the user documentation and internal
 * comments to the code, the above Disclaimer and U.S. Government End
 * Users Notice.
 */

#include "cuLinearSolver.hpp"

using namespace Eigen;

int cuLinearSolver( double *d_A, // Device address of matrix A 
                    const int64_t n,   // rows of A (square matrix)
                    double *d_B, // Device address of matrix B and output of solution
                    const int64_t nrhs // cols of B
                    )
{
    cusolverDnHandle_t cusolverH = NULL;

    using data_type = double;

    const int64_t m = n;
    const int64_t lda = n;
    const int64_t ldb = n;
    Vector<int64_t, -1> Ipiv(n);
    int info = 0;

    int64_t *d_Ipiv = nullptr; /* pivoting sequence */
    int *d_info = nullptr;     /* error info */

    size_t d_lwork = 0;     /* size of workspace */
    void *d_work = nullptr; /* device workspace for getrf */
    size_t h_lwork = 0;     /* size of workspace */
    void *h_work = nullptr; /* host workspace for getrf */

    /* step 1: create cusolver handle, bind a stream */
    CUSOLVER_CHECK(cusolverDnCreate(&cusolverH));

    /* Create advanced params */
    cusolverDnParams_t params;
    CUSOLVER_CHECK(cusolverDnCreateParams(&params));
    CUSOLVER_CHECK(cusolverDnSetAdvOptions(params, CUSOLVERDN_GETRF, CUSOLVER_ALG_0));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_Ipiv), sizeof(int64_t) * Ipiv.size()));
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_info), sizeof(int)));

    /* step 3: query working space of getrf */
    CUSOLVER_CHECK(
        cusolverDnXgetrf_bufferSize(cusolverH, params, n, n, traits<data_type>::cuda_data_type, d_A,
                                    lda, traits<data_type>::cuda_data_type, &d_lwork, &h_lwork));

    CUDA_CHECK(cudaMalloc(reinterpret_cast<void **>(&d_work), sizeof(data_type) * d_lwork));

    /* step 4: LU factorization */
    CUSOLVER_CHECK(cusolverDnXgetrf(cusolverH, params, n, n, traits<data_type>::cuda_data_type,
                                    d_A, lda, d_Ipiv, traits<data_type>::cuda_data_type, d_work,
                                    d_lwork, h_work, h_lwork, d_info));

    CUDA_CHECK(cudaMemcpy(Ipiv.data(), d_Ipiv, sizeof(int64_t) * Ipiv.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));

    if (0 > info) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    CUSOLVER_CHECK(cusolverDnXgetrs(cusolverH, params, CUBLAS_OP_N, n, nrhs, /* nrhs */
                                    traits<data_type>::cuda_data_type, d_A, lda, d_Ipiv,
                                    traits<data_type>::cuda_data_type, d_B, ldb, d_info));

    CUDA_CHECK(cudaFree(d_Ipiv));
    CUDA_CHECK(cudaFree(d_info));
    CUDA_CHECK(cudaFree(d_work));

    CUSOLVER_CHECK(cusolverDnDestroyParams(params));

    CUSOLVER_CHECK(cusolverDnDestroy(cusolverH));

    return EXIT_SUCCESS;
}