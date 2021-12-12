2DConvolution.cu and 3DConvolution.cu files are the naive versions of the convolution algorithm shared in the PolyBench benchmark. 
To execute these naive versions and obtain results:
  -chmod +x 2D-3D_conv_naives.sh
  -./2D-3D_conv_naives.sh
This script will run these convolution algorithms and print out some results for different problem sizes. The below screenshot reveals an example run and corresponding results.

![Screenshot from 2021-12-12 19-36-42](https://user-images.githubusercontent.com/73446582/145721253-82aeb0f9-e1ac-436e-a6eb-c03c57c437b8.png)

As one can simply realize that there is a linear relationship between execution times and problem sizes. 
