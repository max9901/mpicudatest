cmake_minimum_required(VERSION 3.10)
project(testmpicuda LANGUAGES CUDA CXX C)
set(CMAKE_EXPORT_COMPILE_COMMANDS "YES")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --pedantic -Wall -Ofast")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

include(FindCUDA)
find_package(CUDA REQUIRED)
enable_language(CUDA)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

add_executable(testmpicuda main.cpp)
