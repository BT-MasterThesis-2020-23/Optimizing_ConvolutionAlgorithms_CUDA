2DConvolution.cu and 3DConvolution.cu files are the naive versions of the convolution algorithm shared in the PolyBench benchmark. 
To execute these naive versions and obtain results:
  - chmod +x 2D-3D_conv_naives.sh
  - ./2D-3D_conv_naives.sh

This script will run these convolution algorithms and print out some results for different problem sizes. The below screenshot reveals an example run and corresponding results.

![Screenshot from 2021-12-12 19-36-42](https://user-images.githubusercontent.com/73446582/145721253-82aeb0f9-e1ac-436e-a6eb-c03c57c437b8.png)

As one can simply realize that there is a linear relationship between execution times and problem sizes. 

The .ncu files are the profiler executables for the output of 2D-3DConvolution algorithms. Also, I have shared the profiling results. 3D convolution is implemented such that it repeates multiple times of 2DConvolution. Hence, optimizing 2DConvolution with some sophisticated methods will also result in optimization on 3DConvolution.

As profiler results confirm, GPU compute units (SMs) throughput is %50 on average. Suppose that I optimize memory traffic of the SMs by benefitting from the shared memory. In that case, I think that utilization of compute units will increase, increasing overall performance. 

1) Both L1 and L2 hit rates are above %70. 
2) All memory requests communicate and operate with the global memory. However, this memory management can be optimized such that masking 2D-array can be stored in the constant memory/shared memory and, repeated global memory requests for the same data by different threads can be serviced from shared memory instead of global memory.
3) I will try to re-optimize by decreasing the register amount for each thread, and register values that are common for threads in the kernel will be accessed by constant memory. This can enhance performance such that if some of the threads cannot be launched because of the insufficiency of the registers, resultant released registers will be used by those threads, and occupancy will increase in this way. If this would be so, this also increases performance.
4) Also different threadsPerBlock and blocksPerGrid configurations will be tried to increase performance for 2D-3DConvolution algorithms. 
5) I will try to decrease non-coalesced global memory accesses. 
