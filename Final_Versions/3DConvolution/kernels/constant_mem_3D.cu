/* CONSTANT MEMORY - 3D CONVOLUTION */ 

#include <stdio.h>
#include <string.h>
#include <cuda.h>

typedef float DATA_TYPE;

__constant__ DATA_TYPE c[3][3] = {{2, -3, 4}, {5, 6, 7}, {-8, -9, 10}};

__global__ void const_mem_conv3D_kernel(DATA_TYPE *A, DATA_TYPE *B, int i, const int NI)
{
	const int NJ = NI;
	const int NK = NI;	

	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	if ((i < (NI-1)) && (j < (NJ-1)) &&  (k < (NK-1)) && (i > 0) && (j > 0) && (k > 0))
	{
		B[i*(NK * NJ) + j*NK + k] = c[0][0] * (A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)] +  
									A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]) +
								    c[0][2] * (A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)] +
									A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k + 1)]) +
									c[1][0] * (A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k - 1)] +
									A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]) +
									c[1][2] * (A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k - 1)] + 
									A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k + 1)]) + 
									c[2][0] * (A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k - 1)] +
									A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k + 1)]) +
									c[2][2] * (A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k - 1)] + 
									A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k + 1)]) +
									c[0][1] * A[(i + 0)*(NK * NJ) + (j - 1)*NK + (k + 0)] + 
									c[1][1] * A[(i + 0)*(NK * NJ) + (j + 0)*NK + (k + 0)] +
									c[2][1] * A[(i + 0)*(NK * NJ) + (j + 1)*NK + (k + 0)] ;
	}
}

void constant_mem_3D(DATA_TYPE* A, DATA_TYPE* B, const int b_dim, const int mat_dim)
{
	const int NJ = mat_dim;
	const int NK = mat_dim;	

	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * mat_dim * NJ * NK);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * mat_dim * NJ * NK);

	if ((A_gpu == NULL) || (B_gpu == NULL))
		printf("allocation error on the device side\n");

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

	err = cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * mat_dim * mat_dim * mat_dim, cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
		printf("Error for cudaMemcpy : %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);       
	}
	
	dim3 block(b_dim, b_dim, 1);
	dim3 grid((size_t)(ceil(((float)NK) / ((float)block.x))), 
			  (size_t)(ceil(((float)NJ) / ((float)block.y))));

	int i;
	cudaEventRecord(start);
	for (i = 1; i < mat_dim - 1; ++i) // 0
	{
		const_mem_conv3D_kernel<<< grid, block >>>(A_gpu, B_gpu, i, mat_dim);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	err = cudaMemcpy(B, B_gpu, sizeof(DATA_TYPE) * mat_dim * mat_dim * mat_dim, cudaMemcpyDeviceToHost);
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
