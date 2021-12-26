/* CONSTANT_SHARED_MEM_2D_H */

#include <stdio.h>
#include <string.h>
#include <cuda.h>

typedef float DATA_TYPE;

__constant__ DATA_TYPE c_2[3][3] = {{0.2, -0.3, 0.4}, {0.5, 0.6, 0.7}, {-0.8, -0.9, 0.1}};

/*	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;
	c11 = +0.2;  c21 = +0.5;  c31 = -0.8;
	c12 = -0.3;  c22 = +0.6;  c32 = -0.9;
	c13 = +0.4;  c23 = +0.7;  c33 = +0.10;
*/

__global__ void constant_shared_mem_kernel(DATA_TYPE *A, DATA_TYPE *B, int mat_dim, const int b_dim)
{
	__shared__ float shmem[34][34];

    short int gl_ty = blockIdx.x * blockDim.x + threadIdx.x;
    short int gl_tx = blockIdx.y * blockDim.y + threadIdx.y;

    short int lcl_ty = threadIdx.x;
    short int lcl_tx = threadIdx.y;

    if ((gl_ty > 0) && (gl_tx > 0) && (gl_tx < mat_dim -1) && (gl_ty < mat_dim -1))
    {
        shmem[lcl_tx + 1][lcl_ty + 1] = A[gl_tx * mat_dim + gl_ty];

        if(lcl_ty == 0)
            shmem[lcl_tx + 1][0] = A[gl_tx * mat_dim + gl_ty - 1];

        if(lcl_tx == 0)
            shmem[0][lcl_ty + 1] = A[(gl_tx - 1) * mat_dim + gl_ty];

        if(lcl_ty == (b_dim - 1))
            shmem[lcl_tx + 1][b_dim + 1] = A[gl_tx * mat_dim + gl_ty + 1];

        if(lcl_tx == (b_dim - 1))
            shmem[b_dim + 1][lcl_ty + 1] = A[(gl_tx + 1) * mat_dim + gl_ty];
        __syncthreads();

        B[gl_tx * mat_dim + gl_ty] = c_2[0][0] * shmem[lcl_tx][lcl_ty] +
                                    c_2[1][0] * shmem[lcl_tx][lcl_ty + 1] + 
                                    c_2[2][0] * shmem[lcl_tx][lcl_ty + 2] + 
                                    c_2[0][1] * shmem[lcl_tx + 1][lcl_ty ] + 
                                    c_2[1][1] * shmem[lcl_tx + 1][lcl_ty + 1] + 
                                    c_2[2][1] * shmem[lcl_tx + 1][lcl_ty + 2] + 
                                    c_2[0][2] * shmem[lcl_tx + 2][lcl_ty ] + 
                                    c_2[1][2] * shmem[lcl_tx + 2][lcl_ty + 1] + 
                                    c_2[2][2] * shmem[lcl_tx + 2][lcl_ty + 2];

    }
}

void constant_shared_mem_2D(DATA_TYPE* A, DATA_TYPE* B_dev, const int b_dim, int mat_dim)
{
    dim3 threadsPerBlock((size_t)b_dim, (size_t)b_dim, 1);
    dim3 blocksPerGrid((size_t)ceil(mat_dim/b_dim), size_t(ceil(mat_dim/b_dim)), 1);
        
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

   	cudaEvent_t start, stop;
    err = cudaEventCreate(&start);
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
	constant_shared_mem_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_gpu, B_gpu, mat_dim, b_dim);
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





