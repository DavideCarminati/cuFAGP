cmake_minimum_required(VERSION 3.18.0)
project(parallel LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
set(CMAKE_BUILD_TYPE Release)
add_compile_options( -O3 -w )

include(CTest)
enable_testing()

find_package(CUDA)
find_package(OpenMP)
find_package(Eigen3 3.4 REQUIRED NO_MODULE)

include_directories(include/
                    ${Eigen3_DIR}
                    ${PYTHON_INCLUDE_DIRS})

link_directories(${EIGEN_LIBRARY_DIRS})
add_definitions(${EIGEN_DEFINITIONS})

add_executable(gp_cpu src/GP_nD_CPU.cpp)
add_executable(gp_gpu src/GP_nD_GPU.cu src/cuLinearSolver.cu src/cuFAGPutils.cu)

get_target_property(MAIN_CFLAGS gp_gpu COMPILE_OPTIONS)

target_compile_options( gp_gpu PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        #  --generate-line-info # for profiling
                        --use_fast_math
                        -gencode arch=compute_61,code=sm_61
                        -diag-suppress 20012 # Suppressing Eigen warnings
                        # -g
                        # -G
>)

target_link_libraries(  gp_cpu
                        Eigen3::Eigen
                        OpenMP::OpenMP_CXX
                        )

target_link_libraries(  gp_gpu
                        Eigen3::Eigen
                        OpenMP::OpenMP_CXX
                        cublas
                        cusolver
                        cusparse
                        )

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
