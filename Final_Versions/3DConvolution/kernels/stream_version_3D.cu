/*STREAM_VERSION_3D_CONVOLUTION*/

#include <stdio.h>
#include <string.h>
#include <cuda.h>

typedef float DATA_TYPE;

__global__ void stream_conv3D_kernel(DATA_TYPE *A, DATA_TYPE *B, const int mat_dim, 
                                     const short i)
{
	int k = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;

	DATA_TYPE c11, c12, c13, c21, c22, c23, c31, c32, c33;

	c11 = +2;  c21 = +5;  c31 = -8;
	c12 = -3;  c22 = +6;  c32 = -9;
	c13 = +4;  c23 = +7;  c33 = +10;

	if ((i < (mat_dim-1)) && (j < (mat_dim-1)) &&  (k < (mat_dim-1)) && (i > 0) && (j > 0) && (k > 0))
	{
		B[i*(mat_dim * mat_dim) + j*mat_dim + k] = 
                c11 * (A[(i - 1)*(mat_dim * mat_dim) + (j - 1)*mat_dim + (k - 1)] +  
          				A[(i - 1)*(mat_dim * mat_dim) + (j - 1)*mat_dim + (k + 1)]) +
			    c13 * (A[(i + 1)*(mat_dim * mat_dim) + (j - 1)*mat_dim + (k - 1)] +
		                A[(i + 1)*(mat_dim * mat_dim) + (j - 1)*mat_dim + (k + 1)]) +
				c21 * (A[(i - 1)*(mat_dim * mat_dim) + (j + 0)*mat_dim + (k - 1)] +
   						A[(i - 1)*(mat_dim * mat_dim) + (j + 0)*mat_dim + (k + 1)]) +
				c23 * (A[(i + 1)*(mat_dim * mat_dim) + (j + 0)*mat_dim + (k - 1)] + 
        				A[(i + 1)*(mat_dim * mat_dim) + (j + 0)*mat_dim + (k + 1)]) + 
				c31 * (A[(i - 1)*(mat_dim * mat_dim) + (j + 1)*mat_dim + (k - 1)] +
		        		A[(i - 1)*(mat_dim * mat_dim) + (j + 1)*mat_dim + (k + 1)]) +
				c33 * (A[(i + 1)*(mat_dim * mat_dim) + (j + 1)*mat_dim + (k - 1)] + 
		        		A[(i + 1)*(mat_dim * mat_dim) + (j + 1)*mat_dim + (k + 1)]) +
    			c12 * A[(i + 0)*(mat_dim * mat_dim) + (j - 1)*mat_dim + (k + 0)] + 
				c22 * A[(i + 0)*(mat_dim * mat_dim) + (j + 0)*mat_dim + (k + 0)] +
				c32 * A[(i + 0)*(mat_dim * mat_dim) + (j + 1)*mat_dim + (k + 0)] ;
	}
}

void stream_version_3D(DATA_TYPE* A, DATA_TYPE* B_dev, const int b_dim, 
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
            stream_conv3D_kernel<<<blocksPerGrid, threadsPerBlock, 0, streams[stream]>>>
                            (A_gpu, B_gpu, mat_dim, j + 
                            (stream * (mat_dim/num_streams)));
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