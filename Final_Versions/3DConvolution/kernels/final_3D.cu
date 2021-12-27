/* Final Version 3D convolution*/

#include <stdio.h>
#include <string.h>
#include <cuda.h>

typedef float DATA_TYPE;

__constant__ DATA_TYPE c_f[3][3] = {{2, -3, 4}, {5, 6, 7}, {-8, -9, 10}};

__global__ void final_version_3Dkernel(DATA_TYPE *A, DATA_TYPE *B, const int mat_dim, 
                                      const short int b_dim, const short i)
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
                                 c_f[0][0] * (shmem[0][lcl_tx][lcl_ty] + shmem[0][lcl_tx][lcl_ty+2]) +
                                 c_f[1][0] * (shmem[0][lcl_tx+1][lcl_ty] + shmem[0][lcl_tx+1][lcl_ty+2]) + 
                                 c_f[2][0] * (shmem[0][lcl_tx+2][lcl_ty] + shmem[0][lcl_tx+2][lcl_ty+2]) + 
                                 c_f[0][1] * shmem[1][lcl_tx][lcl_ty+1] + 
                                 c_f[1][1] * shmem[1][lcl_tx+1][lcl_ty+1] + 
                                 c_f[2][1] * shmem[1][lcl_tx+2][lcl_ty+1] + 
                                 c_f[0][2] * (shmem[2][lcl_tx][lcl_ty] + shmem[2][lcl_tx][lcl_ty+2]) + 
                                 c_f[1][2] * (shmem[2][lcl_tx+1][lcl_ty] + shmem[2][lcl_tx+1][lcl_ty+2]) + 
                                 c_f[2][2] * (shmem[2][lcl_tx+2][lcl_ty] + shmem[2][lcl_tx+2][lcl_ty+2]);
    }
}


void final_version_3D(DATA_TYPE* A, DATA_TYPE* B_dev, const int b_dim, 
                        const int mat_dim, const int num_streams)
{
    if (num_streams >= 32)
    {
        printf("please enter stream number below than 32\n");
        exit(1);
    }

    dim3 threadsPerBlock(b_dim, b_dim, 1);
    dim3 blocksPerGrid(size_t(mat_dim/b_dim), size_t(mat_dim/b_dim),  1);

   	DATA_TYPE *A_gpu;
	DATA_TYPE *B_gpu;

    cudaError_t err = cudaMalloc((void **)&A_gpu, sizeof(DATA_TYPE) * mat_dim
                                 * mat_dim * mat_dim);
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
    float milliseconds = 0;
    
    cudaStream_t streams[num_streams];

    const int chunk_size = ceil((mat_dim * mat_dim * mat_dim)/num_streams);
    for (int i = 0; i < num_streams; i++)
    {
        err = cudaStreamCreate(&streams[i]);
        if (err != cudaSuccess)
        {
            printf("Error for cudaStreamCreate : %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);       
        }
    }

    for(int stream = 0; stream < num_streams; stream++) 
    {
        const int lower = chunk_size * stream;
        const int upper = min(lower + chunk_size, mat_dim * mat_dim * mat_dim);
        const int width = upper - lower;
        cudaEvent_t start, stop;
        float x = 0;
        err = cudaEventCreate(&start);
        if (err != cudaSuccess)
        {
            printf("Error for cudaEventCreate : %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);       
        }
        err = cudaEventCreate(&stop);
        if (err != cudaSuccess)
        {
            printf("Error for cudaEventCreate : %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);       
        }
        if ((stream != 0) && (stream != num_streams - 1))
            err = cudaMemcpyAsync(A_gpu + lower - mat_dim * mat_dim,
                                  A + lower - mat_dim * mat_dim,
                                  sizeof(DATA_TYPE) * (width + 2 * mat_dim * mat_dim),
                                  cudaMemcpyHostToDevice, streams[stream]);
        else if (stream == 0)
            err = cudaMemcpyAsync(A_gpu, 
                                  A, 
                                  sizeof(DATA_TYPE) * width + mat_dim * mat_dim, 
                                  cudaMemcpyHostToDevice, streams[stream]);

        else if (stream == num_streams - 1)
            err = cudaMemcpyAsync(A_gpu + lower, 
                                  A + lower, 
                                  sizeof(DATA_TYPE) * width, 
                                  cudaMemcpyHostToDevice, streams[stream]);

        if (err != cudaSuccess)
        {
            printf("A\n");
            printf("Error for cudaMemcpyAsync : %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);       
        }
        for (int j = 0; j < mat_dim/num_streams; j++)
        {   
            x = 0;
            cudaEventRecord(start);
            final_version_3Dkernel<<<blocksPerGrid, threadsPerBlock, 0, streams[stream]>>>
                            (A_gpu, B_gpu, mat_dim, b_dim, 
                            j + (stream * (mat_dim/num_streams)));
            cudaEventRecord(stop);
            err = cudaGetLastError();
            if (err != cudaSuccess)
            {
                printf("Kernel\n");
                printf("Error for kernel : %s\n", cudaGetErrorString(err));
                exit(EXIT_FAILURE);       
            }
            cudaEventSynchronize(stop);
            cudaEventElapsedTime(&x, start, stop);
            milliseconds += x;
        }
        err = cudaMemcpyAsync(B_dev + lower, 
                              B_gpu + lower, 
                              sizeof(DATA_TYPE) * width, 
                              cudaMemcpyDeviceToHost, 
                              streams[stream]);
        if (err != cudaSuccess)
        {
            printf("B\n");
            printf("Error for cudaMemcpyAsync : %s\n", cudaGetErrorString(err));
            exit(EXIT_FAILURE);       
        }

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