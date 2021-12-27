nvcc main.cu -o res

echo "problem size = 4096*4096, TB == 32*32 and stream = 8"
./res --naive 4096 32 0 --cpu_time
./res --constant_mem 4096 32 0 --cpu_time
./res --shared_mem 4096 32 0 --cpu_time
./res --stream_version 4096 32 8 --cpu_time
./res --constant_shared_mem 4096 32 0 --cpu_time
./res --stream_const 4096 32 8 --cpu_time
./res --final 4096 32 8 --cpu_time

echo " "
echo "problem size = 8192*8192, TB == 32*32 and stream = 8"
./res --naive 8192 32 0 --cpu_time
./res --constant_mem 8192 32 0 --cpu_time
./res --shared_mem 8192 32 0 --cpu_time
./res --stream_version 8192 32 8 --cpu_time
./res --constant_shared_mem 8192 32 0 --cpu_time
./res --stream_const 8192 32 8 --cpu_time
./res --final 8192 32 8 --cpu_time

echo " "
echo "problem size = 16384*16384, TB == 32*32 and stream = 8"
./res --naive 16384 32 0 --cpu_time
./res --constant_mem 16384 32 0 --cpu_time
./res --shared_mem 16384 32 0 --cpu_time
./res --stream_version 16384 32 8 --cpu_time
./res --constant_shared_mem 16384 32 0 --cpu_time
./res --stream_const 16384 32 8 --cpu_time
./res --final 16384 32 8 --cpu_time

echo " "
echo "problem size = 4096*4096, TB == 16*16 and stream = 8"
./res --naive 4096 16 0 --cpu_time
./res --constant_mem 4096 16 0 --cpu_time
./res --shared_mem 4096 16 0 --cpu_time
./res --stream_version 4096 16 8 --cpu_time
./res --constant_shared_mem 4096 16 0 --cpu_time
./res --stream_const 4096 16 8 --cpu_time
./res --final 4096 16 8 --cpu_time

echo " "
echo "problem size = 4096*4096 and TB == 8*8, stream = 8"
./res --naive 4096 8 0 --cpu_time
./res --constant_mem 4096 8 0 --cpu_time
./res --shared_mem 4096 8 0 --cpu_time
./res --stream_version 4096 8 8 --cpu_time
./res --constant_shared_mem 4096 8 0 --cpu_time
./res --stream_const 4096 8 8 --cpu_time
./res --final 4096 8 8 --cpu_time

echo " "
echo "problem size = 4096*4096, TB == 32*32 and stream = 16"
./res --stream_version 4096 32 16 --cpu_time
./res --stream_const 4096 32 16 --cpu_time
./res --final 4096 32 16 --cpu_time

echo " "
echo "problem size = 4096*4096, TB == 32*32 and stream = 32"
./res --stream_version 4096 32 32 --cpu_time
./res --stream_const 4096 32 32 --cpu_time
./res --final 4096 32 32 --cpu_time
