
/*STREAM_VERSION_2D_CONVOLUTION*/

#include <stdio.h>
#include <string.h>
#include <cuda.h>

typedef float DATA_TYPE;

__global__ void stream_version_kernel(DATA_TYPE *A, DATA_TYPE *B, const int problem_size, const int stream)
{
    int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

    if ((i > 0) && (i < (problem_size/stream) - 1) && (j > 0) && (j < problem_size - 1))
	{
		B[i * problem_size + j] =  c11 * A[(i - 1) * problem_size + (j - 1)] + 
							       c21 * A[(i - 1) * problem_size + (j + 0)] + 
								   c31 * A[(i - 1) * problem_size + (j + 1)] +
			 				       c12 * A[(i + 0) * problem_size + (j - 1)] + 
								   c22 * A[(i + 0) * problem_size + (j + 0)] +
								   c32 * A[(i + 0) * problem_size + (j + 1)] +
			 					   c13 * A[(i + 1) * problem_size + (j - 1)] + 
								   c23 * A[(i + 1) * problem_size + (j + 0)] +
								   c33 * A[(i + 1) * problem_size + (j + 1)];
	}
}

void stream_version_2D(DATA_TYPE* A, DATA_TYPE* B_dev, const int b_dim, 
                        const int mat_dim, const int num_streams)
{
    dim3 threadsPerBlock((size_t)(b_dim * b_dim), 1, 1);
    dim3 blocksPerGrid((mat_dim/(b_dim*b_dim)), 
                size_t((mat_dim * mat_dim)/(b_dim * b_dim * num_streams * (mat_dim/(b_dim * b_dim)))), 1);

   	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

    cudaError_t err = cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * mat_dim * mat_dim);
    if(err != cudaSuccess)
    {
        printf("unsuccessful cuda malloc operation for stream_const_mem_2D\n");
        exit(EXIT_FAILURE);       
    }

	err = cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * mat_dim * mat_dim);
    if(err != cudaSuccess)
    {
        printf("unsuccessful cuda malloc operation for stream_const_mem_2D\n");
        exit(EXIT_FAILURE);       
    }

    cudaStream_t streams[num_streams];
    const int chunk_size = ceil((mat_dim * mat_dim)/num_streams);
	float milliseconds = 0;

    for(int i = 0; i < num_streams; i++)
    {
        cudaStreamCreate(&streams[i]);
    }

    for(int stream = 0; stream < num_streams; stream++) 
    {
        const int lower = chunk_size * stream;      
        const int upper = min(lower + chunk_size, mat_dim * mat_dim);
        const int width = upper - lower;

        cudaEvent_t start, stop;
        float x = 0;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);

        cudaMemcpyAsync(A_gpu + lower, A + lower, sizeof(DATA_TYPE) * width, cudaMemcpyHostToDevice, streams[stream]);
		cudaEventRecord(start);
    	stream_version_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[stream]>>>(A_gpu + lower, 
                                                                        B_gpu + lower, mat_dim, num_streams);
		cudaEventRecord(stop);
        cudaMemcpyAsync(B_dev + lower, B_gpu + lower, sizeof(DATA_TYPE) * width, cudaMemcpyDeviceToHost, streams[stream]);
        cudaEventSynchronize(stop);
	    cudaEventElapsedTime(&x, start, stop);

        milliseconds += x;

        if(stream)  
            cudaStreamSynchronize(streams[stream-1]);
    }
    for(int i = 0; i < num_streams; i++)
    {
        cudaStreamDestroy(streams[i]);
    }
	printf("%0.4lf\n", milliseconds);

	cudaFree(A_gpu);
	cudaFree(B_gpu);    
}