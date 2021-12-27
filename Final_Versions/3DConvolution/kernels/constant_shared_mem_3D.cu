/* CONSTANT_SHARED_MEM_3D_H */

#include <stdio.h>
#include <string.h>
#include <cuda.h>

typedef float DATA_TYPE;

__constant__ DATA_TYPE c_cs[3][3] = {{2, -3, 4}, {5, 6, 7}, {-8, -9, 10}};

__global__ void const_shared_mem_conv3D_kernel(DATA_TYPE *A, DATA_TYPE *B, int mat_dim, 
                        const short int b_dim, const short int i)
{
	__shared__ float shmem[3][34][34];

    short int gl_ty = blockIdx.x * blockDim.x + threadIdx.x;
    short int gl_tx = blockIdx.y * blockDim.y + threadIdx.y;

    short int lcl_ty = threadIdx.x;
    short int lcl_tx = threadIdx.y;

    if ((gl_ty > 0) && (gl_tx > 0) && (gl_tx < mat_dim -1) && (gl_ty < mat_dim -1) &&
        (i > 0) && (i < mat_dim -1))
    {
        shmem[0][lcl_tx+1][lcl_ty+1] = A[(i-1) * mat_dim * mat_dim + gl_tx * mat_dim + gl_ty];
        shmem[1][lcl_tx+1][lcl_ty+1] = A[i * mat_dim * mat_dim + gl_tx * mat_dim + gl_ty];
        shmem[2][lcl_tx+1][lcl_ty+1] = A[(i+1) * mat_dim * mat_dim + gl_tx * mat_dim + gl_ty];

        if(lcl_ty == 0)
        {
            shmem[0][lcl_tx + 1][0] = A[(i-1) * mat_dim * mat_dim + gl_tx * mat_dim + gl_ty - 1];
            shmem[2][lcl_tx + 1][0] = A[(i+1) * mat_dim * mat_dim + gl_tx * mat_dim + gl_ty - 1];
            shmem[1][lcl_tx + 1][0] = A[i * mat_dim * mat_dim + gl_tx * mat_dim + gl_ty - 1];
        }

        if(lcl_tx == 0)
        {
            shmem[0][0][lcl_ty + 1] = A[(i-1) * mat_dim * mat_dim + (gl_tx - 1) * mat_dim + gl_ty];
            shmem[2][0][lcl_ty + 1] = A[(i+1) * mat_dim * mat_dim + (gl_tx - 1) * mat_dim + gl_ty];
            shmem[1][0][lcl_ty + 1] = A[i * mat_dim * mat_dim + (gl_tx - 1) * mat_dim + gl_ty];
        }

        if(lcl_ty == (b_dim - 1))
        {
            shmem[0][lcl_tx + 1][b_dim + 1] = A[(i-1) * mat_dim * mat_dim + gl_tx * mat_dim + gl_ty + 1];
            shmem[2][lcl_tx + 1][b_dim + 1] = A[(i+1) * mat_dim * mat_dim + gl_tx * mat_dim + gl_ty + 1];
            shmem[1][lcl_tx + 1][b_dim + 1] = A[i * mat_dim * mat_dim + gl_tx * mat_dim + gl_ty + 1];
        }

        if(lcl_tx == (b_dim - 1))
        {
            shmem[0][b_dim + 1][lcl_ty + 1] = A[(i-1) * mat_dim * mat_dim + (gl_tx + 1) 
                                                * mat_dim + gl_ty];
            shmem[2][b_dim + 1][lcl_ty + 1] = A[(i+1) * mat_dim * mat_dim + (gl_tx + 1) 
                                                * mat_dim + gl_ty];
            shmem[1][b_dim + 1][lcl_ty + 1] = A[i * mat_dim * mat_dim + (gl_tx + 1) 
                                                * mat_dim + gl_ty];
        }

        if ((lcl_tx == (b_dim -1)) && (lcl_ty == (b_dim -1)))
        {
            shmem[0][b_dim + 1][b_dim + 1] = A[(i-1) * mat_dim * mat_dim + (gl_tx + 1) 
                                                * mat_dim + gl_ty + 1];
            shmem[2][b_dim + 1][b_dim + 1] = A[(i+1) * mat_dim * mat_dim + (gl_tx + 1) 
                                                * mat_dim + gl_ty + 1];
            shmem[1][b_dim + 1][b_dim + 1] = A[i * mat_dim * mat_dim + (gl_tx + 1) 
                                                * mat_dim + gl_ty + 1];
        }

        if ((lcl_tx == 0) && (lcl_ty == (b_dim -1)))
        {
            shmem[0][0][b_dim + 1] = A[(i-1) * mat_dim * mat_dim + (gl_tx - 1) 
                                                * mat_dim + gl_ty + 1];
            shmem[2][0][b_dim + 1] = A[(i+1) * mat_dim * mat_dim + (gl_tx - 1) 
                                                * mat_dim + gl_ty + 1];
            shmem[1][0][b_dim + 1] = A[i * mat_dim * mat_dim + (gl_tx - 1) 
                                                * mat_dim + gl_ty + 1];
        }

        if ((lcl_ty == 0) && (lcl_tx == (b_dim -1)))
        {
            shmem[0][b_dim + 1][0] = A[(i-1) * mat_dim * mat_dim + (gl_tx + 1) 
                                                * mat_dim + gl_ty - 1];
            shmem[2][b_dim + 1][0] = A[(i+1) * mat_dim * mat_dim + (gl_tx + 1) 
                                                * mat_dim + gl_ty - 1];
            shmem[1][b_dim + 1][0] = A[i * mat_dim * mat_dim + (gl_tx + 1) 
                                                * mat_dim + gl_ty - 1];
        }

        if ((lcl_ty == 0) && (lcl_tx == 0))
        {
            shmem[0][0][0] = A[(i-1) * mat_dim * mat_dim + (gl_tx - 1) 
                                                * mat_dim + gl_ty - 1];
            shmem[2][0][0] = A[(i+1) * mat_dim * mat_dim + (gl_tx - 1) 
                                                * mat_dim + gl_ty - 1];
            shmem[1][0][0] = A[i * mat_dim * mat_dim + (gl_tx - 1) 
                                                * mat_dim + gl_ty - 1];
        }
        __syncthreads();

        B[i * mat_dim * mat_dim + gl_tx * mat_dim + gl_ty] = 
                                 c_cs[0][0] * (shmem[0][lcl_tx][lcl_ty] + shmem[0][lcl_tx][lcl_ty+2]) +
                                 c_cs[1][0] * (shmem[0][lcl_tx+1][lcl_ty] + shmem[0][lcl_tx+1][lcl_ty+2]) + 
                                 c_cs[2][0] * (shmem[0][lcl_tx+2][lcl_ty] + shmem[0][lcl_tx+2][lcl_ty+2]) + 
                                 c_cs[0][1] * shmem[1][lcl_tx][lcl_ty+1] + 
                                 c_cs[1][1] * shmem[1][lcl_tx+1][lcl_ty+1] + 
                                 c_cs[2][1] * shmem[1][lcl_tx+2][lcl_ty+1] + 
                                 c_cs[0][2] * (shmem[2][lcl_tx][lcl_ty] + shmem[2][lcl_tx][lcl_ty+2]) + 
                                 c_cs[1][2] * (shmem[2][lcl_tx+1][lcl_ty] + shmem[2][lcl_tx+1][lcl_ty+2]) + 
                                 c_cs[2][2] * (shmem[2][lcl_tx+2][lcl_ty] + shmem[2][lcl_tx+2][lcl_ty+2]);
    }
}


void constant_shared_mem_3D(DATA_TYPE* A, DATA_TYPE* B_dev, const int b_dim, int mat_dim)
{
    dim3 threadsPerBlock((size_t)b_dim, (size_t)b_dim, 1);
    dim3 blocksPerGrid((size_t)ceil(mat_dim/b_dim), size_t(ceil(mat_dim/b_dim)), 1);
        
   	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

    cudaError_t err = cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * mat_dim * mat_dim * mat_dim);
    if(err != cudaSuccess)
    {
        printf("unsuccessful cuda malloc operation for stream_const_mem_2D\n");
        exit(EXIT_FAILURE);       
    }

	err = cudaMalloc((void **)&B_gpu, sizeof(DATA_TYPE) * mat_dim * mat_dim * mat_dim);
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

	err = cudaMemcpy(A_gpu, A, sizeof(DATA_TYPE) * mat_dim * mat_dim * mat_dim, 
                    cudaMemcpyHostToDevice);
	if (err != cudaSuccess)
	{
        printf("A\n");
		printf("Error for cudaMemcpy : %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);       
	}

	cudaEventRecord(start);
    for (int i = 1; i < mat_dim - 1; i++)
    {
    	const_shared_mem_conv3D_kernel<<<blocksPerGrid, threadsPerBlock>>>(A_gpu, B_gpu, mat_dim, b_dim, i);
        cudaDeviceSynchronize();
    }
	cudaEventRecord(stop);

    err = cudaGetLastError();
	if (err != cudaSuccess)
	{
        printf("Kernel\n");
		printf("Error for kernel : %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);       
	}

	err = cudaMemcpy(B_dev, B_gpu, sizeof(DATA_TYPE) * mat_dim * mat_dim * mat_dim, 
                    cudaMemcpyDeviceToHost);
	if (err != cudaSuccess)
	{
        printf("B\n");
		printf("Error for cudaMemcpy : %s\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);       
	}

	float milliseconds = 0;
	cudaEventElapsedTime(&milliseconds, start, stop);
	printf("%0.4lf\n", milliseconds);	
	cudaFree(A_gpu);
	cudaFree(B_gpu);
}


