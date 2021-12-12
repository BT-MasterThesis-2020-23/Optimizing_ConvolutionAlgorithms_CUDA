nvcc 2DConvolution.cu -o 2DConv

echo "2DConvolution Naive Version"
./2DConv 0
echo "problem size = 2048*2048"
./2DConv 2048
echo "problem size = 4096*4096"
./2DConv 4096
echo "problem size = 8192*8192"
./2DConv 8192
echo "problem size = 16384*16384"
./2DConv 16384

echo ""
nvcc 3DConvolution.cu -o 3DConv

echo "3DConvolution Naive Version"
echo "problem size = 64*64*64"
./3DConv 64
echo "problem size = 128*128*128"
./3DConv 128
echo "problem size = 256*256*256"
./3DConv 256
echo "problem size = 512*512*512"
./3DConv 512

