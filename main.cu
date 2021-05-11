#include <iostream>
#include <mpi.h>
#include <mpi-ext.h>

#define MPI_CHECK_RETURN(error_code) {                                           \
    if (error_code != MPI_SUCCESS) {                                             \
        char error_string[BUFSIZ];                                               \
        int length_of_error_string;                                              \
        int world_rank;                                                          \
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);                              \
        MPI_Error_string(error_code, error_string, &length_of_error_string);     \
        fprintf(stderr, "%3d: %s\n", world_rank, error_string);                  \
        exit(1);                                                                 \
    }}

#define CUDA_CHECK_RETURN(value) {										\
	cudaError_t _m_cudaStat = value;									\
	if (_m_cudaStat != cudaSuccess) {									\
        int world_rank;                                                          \
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);                              \
        char * name = (char*) malloc (MPI_MAX_PROCESSOR_NAME * sizeof(char));     \
        int name_len;                                                            \
        MPI_Get_processor_name(name, &name_len);                                 \
		fprintf(stderr, "%3d %s: CUDA Error %s at line %d in file %s\n",	\
             world_rank,name,cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
        printf("%3d %s: CUDA Error %s at line %d in file %s\n",		            \
			 world_rank,name,cudaGetErrorString(_m_cudaStat), __LINE__, __FILE__);		\
		if(value == 2) exit(2);                                         \
		exit(1);														\
	} }

int main() {
        int test = 0;
        int N = 100;

        printf("Compile time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT) && MPIX_CUDA_AWARE_SUPPORT
        printf("This MPI library has CUDA-aware support.\n", MPIX_CUDA_AWARE_SUPPORT);
#elif defined(MPIX_CUDA_AWARE_SUPPORT) && !MPIX_CUDA_AWARE_SUPPORT
        printf("This MPI library does not have CUDA-aware support.\n");
#else
        printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */

        printf("Run time check:\n");
#if defined(MPIX_CUDA_AWARE_SUPPORT)
        if (1 == MPIX_Query_cuda_support()) {
        printf("This MPI library has CUDA-aware support.\n");
    } else {
        printf("This MPI library does not have CUDA-aware support.\n");
    }
#else /* !defined(MPIX_CUDA_AWARE_SUPPORT) */
        printf("This MPI library cannot determine if there is CUDA-aware support.\n");
#endif /* MPIX_CUDA_AWARE_SUPPORT */


        MPI_CHECK_RETURN(MPI_Init_thread(NULL, NULL,MPI_THREAD_FUNNELED, &test));
        if(test != MPI_THREAD_FUNNELED){
            std::cout << "Somethings is wrong with the mpi init " << test << " != MPI_THREAD_FUNNELED (1)\n";
        }
        int world_size;
        int world_rank;
        MPI_CHECK_RETURN(MPI_Comm_size(MPI_COMM_WORLD, &world_size));
        MPI_CHECK_RETURN(MPI_Comm_rank(MPI_COMM_WORLD, &world_rank));

        int *a,*check;
        int *d_a;

        // Allocate host memory
        a   = (int*)malloc(sizeof(int) * N);
        check   = (int*)malloc(sizeof(int) * N);

        int offset = N/world_size*world_rank;
        int sizedit = N/world_size;
        // Initialize host arrays

        for(int i = 0; i < N ; i++) {
            a[i] = 0;
            check[i] = i;
        }
        for(int i = offset; i < N/world_size + offset; i++){
            a[i] = i;
        }

        // Allocate device memory
        CUDA_CHECK_RETURN(cudaMalloc((void**)&d_a, sizeof(float) * N));
        
        // Transfer data from host to device memory
        CUDA_CHECK_RETURN(cudaMemcpy(d_a, a, sizeof(float) * N, cudaMemcpyHostToDevice));


        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        MPI_CHECK_RETURN(MPI_Barrier(MPI_COMM_WORLD));
        std::cout << world_rank << " :check the cpu allgather " << a << "\n";
        MPI_CHECK_RETURN(MPI_Allgather(
                &a[offset],                            //sendbuffer
                sizedit,                               //sendcount
                MPI_INT,                               //type
                a,                                     //receivebuffer
                sizedit,                               //recvcount (from any process)
                MPI_INT,                               //type
                MPI_COMM_WORLD));                      //handle


        for(int i = 0; i < N ; i++) {
            if(a[i] != check[i]){
                printf("ERROR");
                exit(0);
            }
        }

        MPI_CHECK_RETURN(MPI_Barrier(MPI_COMM_WORLD));
        std::cout << "\n";
        MPI_CHECK_RETURN(MPI_Barrier(MPI_COMM_WORLD));

        std::cout << world_rank << " : check the GPU allgather " << d_a << "\t"  << "\n";
        MPI_CHECK_RETURN(MPI_Allgather(
                &d_a[offset],                           //sendbuffer
                sizedit,                              //sendcount
                MPI_INT,                              //type
                d_a,                                  //receivebuffer
                sizedit,                              //recvcount (from any process)
                MPI_INT,                              //type
                MPI_COMM_WORLD));                     //handle

        CUDA_CHECK_RETURN(cudaMemcpy(a, d_a, sizeof(float) * N, cudaMemcpyDeviceToHost));
        
        CUDA_CHECK_RETURN(cudaDeviceSynchronize());

        for(int i = 0; i < N ; i++) {
            if(a[i] != check[i]){
                printf("ERROR");
                exit(0);
            }
        }

        MPI_Finalize();

        exit(0);
        return true;
}
