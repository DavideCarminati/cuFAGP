cmake_minimum_required(VERSION 3.18.0)
project(parallel LANGUAGES CXX CUDA)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
set(CMAKE_BUILD_TYPE Release)
add_compile_options( -O3 -w )

include(CTest)
enable_testing()

find_package(CUDA)
find_package(OpenMP)
find_package(Eigen3 3.4.0 REQUIRED NO_MODULE)
find_package(Python3 REQUIRED)

include_directories(include/ 
                    ${Eigen3_DIR}
                    ${PYTHON_INCLUDE_DIRS})

link_directories(${EIGEN_LIBRARY_DIRS})
add_definitions(${EIGEN_DEFINITIONS})

add_executable(gp_cpu GP_nD_CPU.cpp)
add_executable(gp_gpu GP_nD_GPU.cu cuLinearSolver.cu cuFAGPutils.cu)

get_target_property(MAIN_CFLAGS gp_gpu COMPILE_OPTIONS)

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
