export OMP_NUM_THREADS=4
gcc -fopenmp -o train-omp-parallel train_omp_parallel.c -lm