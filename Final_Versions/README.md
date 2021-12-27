  2DConvolution and 3DConvolution folders includes experiments for 2D-3D convolution operations. In each folder, there is a file namely **run_experiment.sh**, and to carry out the experiments, one needs to do following:
  - chmod +x run_experiment.sh
  - ./run_experiment.sh

  2DConvolution includes 7 different experiments and 3DConvolution includes 6 different experiments. In those experiments, there are 7 different kernel configurations that operate on the same data. Those kernels are:
  1) **Naive:** Naive implementation
  2) **Constant memory:** This kernel stores the masking 2D-array into constant memory instead of storing them into registers.
  3) **Shared memory:** This kernel implements shared memory usage for repeating elements to decrease global memory traffic between GPU cores and GPU DRAM.  
  4) **Constant memory + Shared memory:** This kernel uses constant memory and shared memory implementations together which are explained as above.
  5) **Stream version:** This kernel divides generated data into streams and benefits from a-synchronized memory copy operations to decrease time elapsed during data transmission between device and host.
  6)  **Constant memory + Stream version:** This kernel uses stream and constant memory implementations together.
  7)  **Final:** This kernel joins constant memory, shared memory and stream usages to obtain the most optimized versions for 2D and 3D convolution operations.

  Results for each experiment are shared in the .pdf files inside 2D-3DConvolution folders. Since my desktop computer cannot benefit from the shared memory (as mentioned in this link: ), I also carry out experiments on the Google-Colab which uses Tesla-K80 GPUs.
  
  I will share detailed explanations and comparison plots in both the final report and my presentation. 
