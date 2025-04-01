#include <stdio.h>
#include <omp.h>
#include "../libs/csr_utils.h"
#include "../libs/csr_Operations.h"
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
#include <string.h>

double *y_SerialResult = NULL;

/* Funzione per svolgere il prodotto matrice-vettore, con memorizzazione CSR della matrice, in modo serializzato */
struct matrixPerformance serial_csr(struct matrixData *matrix_data, double *x, int num_threads) {
    int *IRP, *JA;
    double *AS;

    /* Vettore di output del risultato y <- Ax */
    double *y = malloc((size_t)matrix_data->M * sizeof(double));
    if (y == NULL) {
        printf("Errore nell'allocazione della memoria per il vettore di output y\n");
        exit(EXIT_FAILURE);
    }

    /* Conversione in formato CSR */
    convert_to_csr(matrix_data->M, matrix_data->nz, matrix_data->row_indices, matrix_data->col_indices, matrix_data->values, &IRP, &JA, &AS);

    double start = omp_get_wtime();
    matvec_csr(matrix_data->M, IRP, JA, AS, x, y);
    double end = omp_get_wtime();

    y_SerialResult = malloc(matrix_data->M * sizeof(double));
    memcpy(y_SerialResult, y, matrix_data->M * sizeof(double));

    struct matrixPerformance node;
    node.seconds = end - start;
    node.flops = 0;
    node.gigaFlops = 0;

    free(y);
    free(IRP);
    free(JA);
    free(AS);

    return node;
}

// Prodotto matrice-vettore in CSR
void matvec_csr(int M, const int *IRP, const int *JA, const double *AS, const double *x, double *y) {
    for (int i = 0; i < M; i++) {
        double sum = 0.0;
        for (int j = IRP[i]; j < IRP[i + 1]; j++) {
            sum += AS[j] * x[JA[j]];
        }
        y[i] = sum;
    }
}