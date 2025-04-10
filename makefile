# Compilatore
CC = gcc
NVCC = nvcc
# Opzioni di compilazione
CFLAGS = -O2 -fopenmp -Ilibs
NVFLAGS = -O2 -std=c++11 -Ilibs -ICUDA_libs  -gencode arch=compute_75,code=sm_75
# File sorgenti
SRC_CPU = src/main.c src/csr_utils.c src/csr_Operations.c src/matrixIO.c \
          src/hll_Operations.c src/hll_ellpack_utils.c
# File sorgenti CUDA
SRC_CUDA = src/CUDA_src/main_cuda.cu \
           src/CUDA_src/csr_Operations.cu \
           src/CUDA_src/csr_utils.cu \
           src/CUDA_src/hll_ellpack_utils.cu \
           src/CUDA_src/hll_Operations.cu
# Output binari
TARGET_CPU = main
TARGET_CUDA = cuda_main

# Target predefinito
all: $(TARGET_CPU)

# Compila codice CPU
$(TARGET_CPU): $(SRC_CPU)
	$(CC) $(CFLAGS) $^ -o $@

# Compila codice CUDA
cuda: $(SRC_CUDA)
	$(NVCC) $(NVFLAGS) $^ -o $(TARGET_CUDA)

# Pulizia base
clean:
	rm -f $(TARGET_CPU) $(TARGET_CUDA)

# Pulizia approfondita
deepclean: clean
	rm -f *.o *.out *.txt *.csv