/* NAIVE_2D_CONVOLUTION*/

#include <stdio.h>
#include <string.h>
#include <cuda.h>

typedef float DATA_TYPE;

__global__ void naive_kernel(DATA_TYPE *A, DATA_TYPE *B, const int problem_size)
{
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int i = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;

	if ((i < problem_size-1) && (j < problem_size-1) && (i > 0) && (j > 0))
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

void naive_2D(DATA_TYPE* A, DATA_TYPE* B_dev, const int b_dim, const int mat_dim)
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

    cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * mat_dim * mat_dim);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * mat_dim * mat_dim);

	if ((A_gpu == NULL) || (B_gpu == NULL))
		printf("allocation error on the device side\n");
	
	dim3 block(b_dim, b_dim, 1);
	dim3 grid(ceil(((float)mat_dim) / ((float)block.x)), ceil( ((float)mat_dim) / ((float)block.y)), 1);

	cudaEvent_t start, stop;

    cudaError_t err = cudaEventCreate(&start);
	if (err != cudaSuccess)
	{	
		printf("Error : %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);       
	}
	err = cudaEventCreate(&stop);
	if (err != cudaSuccess)
	{
		printf("Error : %s", cudaGetErrorString(err));
        exit(EXIT_FAILURE);       
	}
	err = cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * mat_dim * mat_dim, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("Error for cudaMemcpy : %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);       
	}
	cudaEventRecord(start);
	naive_kernel<<<grid, block>>>(A_gpu, B_gpu, mat_dim);
	cudaEventRecord(stop);

    err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("Error for cudaMemcpy : %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);       
	}

	err = cudaMemcpy(B_dev, B_gpu, sizeof(DATA_TYPE) * mat_dim * mat_dim, cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
		printf("Error for cudaMemcpy : %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);       
	}
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	printf("%0.4lf\n", milliseconds);
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
}
