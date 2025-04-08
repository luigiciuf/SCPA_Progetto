# Compilatore
CC = gcc

# Opzioni di compilazione
CFLAGS = -O2 -fopenmp -Ilibs

# File sorgenti
SRC = src/main.c src/csr_utils.c src/csr_Operations.c src/matrixIO.c   src/hll_Operations.c \
    src/hll_ellpack_utils.c

# Output finale
TARGET = main

# Regola principale
all: $(TARGET)

# Come costruire l'eseguibile
$(TARGET): $(SRC)
	$(CC) $(CFLAGS) $(SRC) -o $(TARGET)

# Pulizia dei file compilati
clean:
	rm -f $(TARGET)

deepclean: clean
	rm -f *.o *.out *.txt *.csv
	rm -f $(TARGET) cuda_main

# Compilazione CUDA seriale
cuda:
	nvcc -O2 -std=c++11 -Ilibs -ICUDA_libs -o cuda_main \
	src/CUDA_src/main_cuda.cu src/CUDA_src/csr_Operations.cu src/CUDA_src/csr_utils.cu
