/**
 * 3DConvolution.cu: This file is optimized by burak topcu.
 * 
 * Contact: topcuuburak@gmail.com>
 * Teaching / Research Assistant at IZTECH
 * Web address: https://ceng.iyte.edu.tr/people/burak-topcu/
 * 
 * nvcc main.cu -o res
 * 1st argument is experiment type (--naive, --constant_mem, --stream_version etc.)
 * 2nd argument is matrix dimensions (1 dim of square matrix, 64, 128, 256 etc)
 * 3rd argument is thread block dimension size (8, 16, 32)
 * 4th argument is num_streams (2,4,8,16,32)
 * 5th argument cpu time (--cpu_time)
 * 
 */

#include <unistd.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <cuda.h>

#include "kernels/polybenchUtilFuncts.h"
#include "kernels/constant_mem_3D.cu"
#include "kernels/constant_shared_mem_3D.cu"
#include "kernels/final_3D.cu"
#include "kernels/naive_3D.cu"
#include "kernels/shared_mem_3D.cu"
#include "kernels/stream_const_3D.cu"
#include "kernels/stream_version_3D.cu"
#include "kernels/timer.cuh"

#define PERCENT_DIFF_ERROR_THRESHOLD 0.05
#define GPU_DEVICE 0

typedef float DATA_TYPE;

void conv3D(DATA_TYPE* A, DATA_TYPE* B, const int mat_dim)
{
	int i, j, k;

	const int NJ = mat_dim;
	const int NK = mat_dim;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	for (i = 1; i < mat_dim - 1; ++i) // 0
	{
		for (j = 1; j < NJ - 1; ++j) // 1
		{
			for (k = 1; k < NK -1; ++k) // 2
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
	}
}

void init(DATA_TYPE* A, const int mat_dim)
{
	int i, j, k;

	const int NJ = mat_dim;
	const int NK = mat_dim;

	for (i = 0; i < mat_dim; ++i)
   	{
		for (j = 0; j < NJ; ++j)
		{
			for (k = 0; k < NK; ++k)
			{
				A[i*(NK * NJ) + j*NK + k] = (float)rand()/RAND_MAX;
			}
		}
	}
}


void compareResults(DATA_TYPE* B, DATA_TYPE* B_Gpu, const int mat_dim)
{
	const int NJ = mat_dim;
	const int NK = mat_dim;	

	int i, j, k, fail;
	fail = 0;

	for (i = 1; i < mat_dim - 1; ++i) 
	{
		for (j = 1; j < NJ - 1; ++j) 
		{
			for (k = 1; k < NK - 1; ++k) 
			{
				if (percentDiff(B[i*(NK * NJ) + j*NK + k], B_Gpu[i*(NK * NJ) + j*NK + k]) 
								> PERCENT_DIFF_ERROR_THRESHOLD)
				{
					fail++;
				}
			}	
		}
	}
	printf("Failure in percent: %0.4f\n",
		(float)((float)100*((float)fail))/((float)(mat_dim * mat_dim * mat_dim)));
}

void GPU_argv_init()
{
	cudaDeviceProp deviceProp;
	cudaGetDeviceProperties(&deviceProp, GPU_DEVICE);
	printf("setting device %d with name %s\n",GPU_DEVICE,deviceProp.name);
    printf("Max Threads Per block %d\n", deviceProp.maxThreadsPerBlock);
    printf("Max Threads Dim (x, y, z) = (%d, %d, %d)\n", deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
    printf("Max Threads Grid Size (x, y, z) = (%d, %d, %d)\n", deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
	cudaSetDevice( GPU_DEVICE );
}

int main(int argc, char *argv[])
{	
	int block_size, mat_dim, num_streams;

	Timer timer;
	if (argv[2] != NULL)
		mat_dim = atoi(argv[2]);
	
	if (argv[3] != NULL)
		block_size = atoi(argv[3]);
	
	if (argv[4] != NULL)
		num_streams = atoi(argv[4]);

	DATA_TYPE* A;
	DATA_TYPE* B;  
	DATA_TYPE* B_dev;

    cudaMallocHost((void **)&A, sizeof(DATA_TYPE) * mat_dim * mat_dim * mat_dim); 
    cudaMallocHost((void **)&B_dev, sizeof(DATA_TYPE) * mat_dim * mat_dim * mat_dim); 

	B = (DATA_TYPE*)malloc(mat_dim * mat_dim * mat_dim * sizeof(DATA_TYPE));

	if ((A == NULL) || (B == NULL) || (B_dev == NULL))
	{
		printf("Allocation error on the host side\n");
	}

	init(A, mat_dim);

	if (argv[1] != NULL)
	{
		printf("Matrix dim = %dx%dx%d, block_size = %dx%d, streams = %d\n", 
		mat_dim, mat_dim, mat_dim, block_size, block_size, num_streams);
        timer.start();
		if(strcmp(argv[1], "--naive") == 0)
		{
			printf("Naive version\n");
			naive_3D(A, B_dev, block_size, mat_dim);
		}
		if(strcmp(argv[1], "--constant_mem") == 0)
		{
			printf("With constant memory\n");
			constant_mem_3D(A, B_dev, block_size, mat_dim);
		}
		if(strcmp(argv[1], "--constant_shared_mem") == 0)
		{
			printf("With constant and shared memory\n");
			constant_shared_mem_3D(A, B_dev, block_size, mat_dim);
		}
		if(strcmp(argv[1], "--shared_mem") == 0)
		{	
			printf("With shared memory\n");
			shared_mem_3D(A, B_dev, block_size, mat_dim);
		}
		if(strcmp(argv[1], "--stream_version") == 0)
		{	
			printf("With stream implemented\n");
			stream_version_3D(A, B_dev, block_size, mat_dim, num_streams);
		}
		if(strcmp(argv[1], "--stream_const") == 0)
		{
			printf("With stream and constant memory implementation\n");
			stream_const_3D(A, B_dev, block_size, mat_dim, num_streams);
		}
		if(strcmp(argv[1], "--final") == 0)
		{
			printf("Final version\n");
			final_version_3D(A, B_dev, block_size, mat_dim, num_streams);
		}
		timer.stop("");
	}

	if (argv[5] != NULL)
	{
		if (strcmp(argv[5], "--cpu_time") == 0)
		{
			double t_start, t_end;

			t_start = rtclock();
			conv3D(A, B, mat_dim);
			t_end = rtclock();
			fprintf(stdout, "%0.6lf\n", t_end - t_start);

			compareResults(B, B_dev, mat_dim);
		}
	}

    cudaFreeHost(A);    
    cudaFreeHost(B_dev);
	free(B);
	
	return 0;
}




