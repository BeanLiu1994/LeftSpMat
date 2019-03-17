cmake_minimum_required(VERSION 2.8)

project(spMat_test)

find_package(CUDA REQUIRED)
find_package(Eigen3 REQUIRED)

include_directories(${EIGEN3_INCLUDE_DIR})

file(GLOB PATHS_SRC *.cu *.cpp)
CUDA_ADD_EXECUTABLE(spMat_test ${PATHS_SRC})