cmake_minimum_required(VERSION 3.16)
project(testmpicuda)

set(CMAKE_CXX_STANDARD 14)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread --pedantic -Wall -O3 -march=native")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS})

find_package(MPI REQUIRED)
include(FindCUDA)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 11)
include_directories(SYSTEM ${MPI_INCLUDE_PATH})
message("[sim HH] MPI enabled")
EXECUTE_PROCESS (COMMAND "ompi_info" --parsable --all COMMAND "grep" "mpi_built_with_cuda_support:value" OUTPUT_VARIABLE OMPI_CUDA_OUTPUT)

add_executable(testmpicuda main.cu)

target_link_libraries(testmpicuda PRIVATE ${CUDA_LIBRARIES})
target_link_libraries(testmpicuda PUBLIC  ${CUDA_LIBRARIES})
target_link_libraries(testmpicuda PRIVATE ${MPI_CXX_LIBRARIES})

