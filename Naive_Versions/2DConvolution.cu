/**
 * 2DConvolution.cu: This file is part of the PolyBench/GPU 1.0 test suite.
 *
 *
 * Contact: Scott Grauer-Gray <sgrauerg@gmail.com>
 * Louis-Noel Pouchet <pouchet@cse.ohio-state.edu>
 * Web address: http://www.cse.ohio-state.edu/~pouchet/software/polybench/GPU
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include "polybenchUtilFuncts.h"

//define the error threshold for the results "not matching"
#define PERCENT_DIFF_ERROR_THRESHOLD 0.05

#define GPU_DEVICE 0

/* Thread block dimensions */
#define DIM_THREAD_BLOCK_X 32
#define DIM_THREAD_BLOCK_Y 8

/* Can switch DATA_TYPE between float and double */
typedef float DATA_TYPE;

void conv2D(DATA_TYPE* A, DATA_TYPE* B, const int NI)
{
	int i, j;
	const int NJ = NI;
	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;


	for (i = 1; i < NI - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			B[i*NJ + j] = c11 * A[(i - 1)*NJ + (j - 1)]  +  c12 * A[(i + 0)*NJ + (j - 1)]  +  c13 * A[(i + 1)*NJ + (j - 1)]
				+ c21 * A[(i - 1)*NJ + (j + 0)]  +  c22 * A[(i + 0)*NJ + (j + 0)]  +  c23 * A[(i + 1)*NJ + (j + 0)] 
				+ c31 * A[(i - 1)*NJ + (j + 1)]  +  c32 * A[(i + 0)*NJ + (j + 1)]  +  c33 * A[(i + 1)*NJ + (j + 1)];
		}
	}
}

void init(DATA_TYPE* A, const int problem_size)
{
	int i, j;

	for (i = 0; i < problem_size; ++i)
    {
		for (j = 0; j < problem_size; ++j)
		{
			A[i * problem_size + j] = (float)rand()/RAND_MAX;
       	}
    }
}


void compareResults(DATA_TYPE* B, DATA_TYPE* B_outputFromGpu, const int problem_size)
{
	int i, j, fail;
	fail = 0;
	
	// Compare a and b
	for (i=1; i < (problem_size-1); i++) 
	{
		for (j=1; j < (problem_size-1); j++) 
		{
			if (percentDiff(B[i*problem_size + j], B_outputFromGpu[i*problem_size + j]) > PERCENT_DIFF_ERROR_THRESHOLD) 
			{
				fail++;
			}
		}
	}
	if (fail != 0)
		printf("Failure: %d\n", fail);	
}


void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("Device name %s\n",deviceProp.name);
	printf("Shared memory per block is: %lu bytes\n",deviceProp.sharedMemPerBlock);
	printf("Total Const mem is: %lu bytes\n",deviceProp.totalConstMem);
	cudaSetDevice( GPU_DEVICE );
}

__global__ void Convolution2D_kernel(DATA_TYPE *A, DATA_TYPE *B, const int problem_size)
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


void convolution2DCuda(DATA_TYPE* A, DATA_TYPE* B, DATA_TYPE* B_outputFromGpu, const int problem_size)
{
	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

	cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * problem_size * problem_size);
	cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * problem_size * problem_size);
	cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * problem_size * problem_size, cudaMemcpyHostToDevice);
	
	dim3 block(DIM_THREAD_BLOCK_X, DIM_THREAD_BLOCK_Y);
	dim3 grid((size_t)ceil( ((float)problem_size) / ((float)block.x) ), (size_t)ceil( ((float)problem_size) / ((float)block.y)) );

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

	cudaEventRecord(start);
	Convolution2D_kernel<<<grid, block>>>(A_gpu, B_gpu, problem_size);
    cudaError_t err = cudaGetLastError();
	cudaEventRecord(stop);
	cudaMemcpy(B_outputFromGpu, B_gpu, sizeof(DATA_TYPE) * problem_size * problem_size, cudaMemcpyDeviceToHost);

	cudaEventSynchronize(stop);
	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);

	fprintf(stdout, "GPU Runtime: %0.6lfms\n", milliseconds);
	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
}

int main(int argc, char *argv[])
{
	int problem_size = atoi(argv[1]);

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* B_outputFromGpu;
	
	A = (DATA_TYPE*)malloc(problem_size*problem_size*sizeof(DATA_TYPE));
	B = (DATA_TYPE*)malloc(problem_size*problem_size*sizeof(DATA_TYPE));
	B_outputFromGpu = (DATA_TYPE*)malloc(problem_size*problem_size*sizeof(DATA_TYPE));

	//initialize the arrays
	init(A, problem_size);
	if (problem_size == 0)
		GPU_argv_init();
	else
	{	
		convolution2DCuda(A, B, B_outputFromGpu, problem_size);	
		conv2D(A, B, problem_size);
		compareResults(B, B_outputFromGpu, problem_size);
	}

	free(A);
	free(B);
	free(B_outputFromGpu);
	return 0;
}

