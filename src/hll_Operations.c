#include <omp.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include "../libs/data_structure.h"
#include "../libs/hll_ellpack_utils.h"
#include "../libs/csr_Operations.h"
#include "../libs/costants.h"



// Funzione semplice per implementare il prodotto matrice-vettore HLL seriale
struct matrixPerformance serial_hll(struct matrixData *matrix_data, double *x_h, int num_threads) {
    int M = matrix_data->M;

    HLL_Matrix *hll_matrix = malloc(sizeof(HLL_Matrix));
    if (!hll_matrix) {
        fprintf(stderr, "Errore: Allocazione fallita per HLL_Matrix.\n");
        exit(EXIT_FAILURE);
    }

    // Calcolo del numero di blocchi
    hll_matrix->num_blocks = (M + HackSize - 1) / HackSize;

    // Allocazione dei blocchi
    hll_matrix->blocks = (ELLPACK_Block *)malloc((size_t)hll_matrix->num_blocks * sizeof(ELLPACK_Block));
    if (!hll_matrix->blocks) {
        fprintf(stderr, "Errore: Allocazione fallita per i blocchi ELLPACK.\n");
        free(hll_matrix);
        exit(EXIT_FAILURE);
    }

    // Conversione in formato HLL
    convert_to_hll(matrix_data, hll_matrix);

    double *y = malloc((size_t)M * sizeof(double));
    if (!y) {
        fprintf(stderr, "Errore: Allocazione fallita per il vettore y.\n");
        free(hll_matrix->blocks);
        free(hll_matrix);
        exit(EXIT_FAILURE);
    }

    double start = omp_get_wtime();
    matvec_Hll_serial(hll_matrix, x_h, y, matrix_data->M);
    double end = omp_get_wtime();

    const double time_spent = end - start;

    struct matrixPerformance performance;
    performance.seconds = time_spent;
    performance.flops = 2.0 * matrix_data->nz;
    performance.gigaFlops = performance.flops / performance.seconds / 1e9;
    
    // Libera memoria
    free(y);
    for (int i = 0; i < hll_matrix->num_blocks; i++) {
        free(hll_matrix->blocks[i].JA);
        free(hll_matrix->blocks[i].AS);
    }
    free(hll_matrix->blocks);
    free(hll_matrix);

    return performance;
}