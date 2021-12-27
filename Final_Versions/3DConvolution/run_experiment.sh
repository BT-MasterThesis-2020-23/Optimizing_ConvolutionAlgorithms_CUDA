nvcc main.cu -o res

echo "data size = 128*128*128, tb size = 16*16, num_stream = 8"
./res --naive 128 16 0 --cpu_time
./res --constant_mem 128 16 0 --cpu_time
./res --shared_mem 128 16 0 --cpu_time
./res --constant_shared_mem 128 16 0 --cpu_time
./res --stream_version 128 16 8 --cpu_time
./res --stream_const 128 16 8 --cpu_time
./res --final 128 16 8 --cpu_time

echo " "
echo "data size = 256*256*256, tb size = 16*16, num_stream = 8"
./res --naive 256 16 0 --cpu_time
./res --constant_mem 256 16 0 --cpu_time
./res --shared_mem 256 16 0 --cpu_time
./res --constant_shared_mem 256 16 0 --cpu_time
./res --stream_version 256 16 8 --cpu_time
./res --stream_const 256 16 8 --cpu_time
./res --final 256 16 8 --cpu_time

echo " "
echo "data size = 512*512*512, tb size = 16*16, num_stream = 8"
./res --naive 512 16 0 --cpu_time
./res --constant_mem 512 16 0 --cpu_time
./res --shared_mem 512 16 0 --cpu_time
./res --constant_shared_mem 512 16 0 --cpu_time
./res --stream_version 512 16 8 --cpu_time
./res --stream_const 512 16 8 --cpu_time
./res --final 512 16 8 --cpu_time

echo " "
echo "data size = 512*512*512, tb size = 32*32, num_stream = 8"
./res --naive 512 32 0 --cpu_time
./res --constant_mem 512 32 0 --cpu_time
./res --shared_mem 512 32 0 --cpu_time
./res --constant_shared_mem 512 32 0 --cpu_time
./res --stream_version 512 32 8 --cpu_time
./res --stream_const 512 32 8 --cpu_time
./res --final 512 32 8 --cpu_time

echo " "
echo "data size = 256*256*256, tb size = 32*32, num_stream = 16"
./res --naive 256 32 0 --cpu_time
./res --constant_mem 256 32 0 --cpu_time
./res --shared_mem 256 32 0 --cpu_time
./res --constant_shared_mem 256 32 0 --cpu_time
./res --stream_version 256 32 16 --cpu_time
./res --stream_const 256 32 16 --cpu_time
./res --final 256 32 16 --cpu_time

echo " "
echo "data size = 256*256*256, tb size = 32*32, num_stream = 4"
./res --naive 256 32 0 --cpu_time
./res --constant_mem 256 32 0 --cpu_time
./res --shared_mem 256 32 0 --cpu_time
./res --constant_shared_mem 256 32 0 --cpu_time
./res --stream_version 256 32 4 --cpu_time
./res --stream_const 256 32 4 --cpu_time
./res --final 256 32 4 --cpu_time
