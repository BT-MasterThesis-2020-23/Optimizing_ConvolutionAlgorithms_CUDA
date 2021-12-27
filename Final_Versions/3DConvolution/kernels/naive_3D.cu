/* NAIVE_3D_CONVOLUTION*/

#include <stdio.h>
#include <string.h>
#include <cuda.h>

typedef float DATA_TYPE;

__global__ void naive_conv3D_kernel(DATA_TYPE *A, DATA_TYPE *B, int i, const int NI)
{
	const int NJ = NI;
	const int NK = NI;	

	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;


	if ((i < (NI-1)) && (j < (NJ-1)) &&  (k < (NK-1)) && (i > 0) && (j > 0) && (k > 0))
	{
		B[i*(NK * NJ) + j*NK + k] = c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k - 1)] +  
									c11 * A[(i - 1)*(NK * NJ) + (j - 1)*NK + (k + 1)] +
								    c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k - 1)] +
									c13 * A[(i + 1)*(NK * NJ) + (j - 1)*NK + (k + 1)] +
									c21 * A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k - 1)] +
									c21 * A[(i - 1)*(NK * NJ) + (j + 0)*NK + (k + 1)] +
									c23 * A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k - 1)] + 
									c23 * A[(i + 1)*(NK * NJ) + (j + 0)*NK + (k + 1)] +
									c31 * A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k - 1)] +
									c31 * A[(i - 1)*(NK * NJ) + (j + 1)*NK + (k + 1)] +
									c33 * A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k - 1)] + 
									c33 * A[(i + 1)*(NK * NJ) + (j + 1)*NK + (k + 1)] +
									c12 * A[(i + 0)*(NK * NJ) + (j - 1)*NK + (k + 0)] + 
									c22 * A[(i + 0)*(NK * NJ) + (j + 0)*NK + (k + 0)] +
									c32 * A[(i + 0)*(NK * NJ) + (j + 1)*NK + (k + 0)] ;
	}
}

void naive_3D(DATA_TYPE* A, DATA_TYPE* B, const int b_dim, const int mat_dim)
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
		naive_conv3D_kernel<<< grid, block >>>(A_gpu, B_gpu, i, mat_dim);
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


