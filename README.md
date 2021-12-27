# CENG443, Heterogeneus Programming
# Final Project

**Proposal:** In this project, I will try to re-implement 2DConvolution and 3DConvolution algorithms from the PolyBench benchmark. One can confirm that implementations for those algorithms are done naively without taking care of the optimizations. I will try to optimize them by adding shared memory usage for convolution operations. In the end, I also provide a performance comparison between naive versions and optimized versions with the help of simulation results taken from the NVIDIA Nsight Compute Tool. 

For further, There are also some other convolution algorithms inside the CUDA-SDK. If the convolution algorithms of PolyBench do not require too much effort, I also want to include convolution-based algorithms implemented in CUDA-SDK in my final project. 

**Final edit:** 2DConvolution and 3DConvolution algorithms implemented in PolyBench benchmark was optimized, related codes and further explanations are shared in the FinalVersion.
