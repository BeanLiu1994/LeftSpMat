cmake_minimum_required(VERSION 2.8)

project(spMat_test)

add_compile_options(-std=c++11)

find_package(CUDA REQUIRED)
list(APPEND CUDA_NVCC_FLAGS "-std=c++11")
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})


file(GLOB PATHS_SRC *.cu *.cpp)
CUDA_ADD_EXECUTABLE(spMat_test ${PATHS_SRC})
