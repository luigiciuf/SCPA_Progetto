

#ifndef DATA_STRUCTURE_H
#define DATA_STRUCTURE_H

#include "mmio.h"
#include <stdint.h>

// Struttura per i dati della matrice
struct matrixData {
    int *row_indices;
    int *col_indices;
    double *values;
    int M;
    int N;
    int nz;
    MM_typecode matcode;
};


// Struttura per le performance
struct matrixPerformance {
    char nameMatrix[50];
    double seconds;
    double flops;
    double gigaFlops;
    double relativeError;
};

struct matrixPerformanceAverage {
    char nameMatrix[50];  // Nome della matrice
    double avarangeFlops; // Flops medi
    double avarangeMegaFlops; // MFLOPS medi
    double avarangeSeconds; // Tempo medio
};

typedef struct {
    char nameMatrix[50];
    double total_seconds;
    int count;
    int nz;
    int row;
    int col;
    double relativeError;
} MatrixPerformanceResult;


typedef struct matrixPerformance (*MatVecKernel)(
    struct matrixData *matrix_data,
    double *x,
    int num_threads
);

struct matrixPerformance benchmark(
    struct matrixData *matrix_data,
    double *x,
    int num_iter,
    int num_threads,
    MatVecKernel kernel_func
);
#endif // DATA_STRUCTURE_H