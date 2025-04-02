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
    node.flops = 2.0 * matrix_data->nz;
    node.gigaFlops = (node.flops / node.seconds) / 1e9;


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
/* Prodotto matrice-vettore parallelo */
void matvec_csr_openMP(const int *IRP, const int *JA, const double *AS, const double *x, double *y, const int *start_row, const int *end_row, int num_threads, int nz, int M) {
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        for (int i = start_row[tid]; i < end_row[tid]; i++) {
            y[i] = 0.0;
            for (int j = IRP[i]; j < IRP[i + 1]; j++) {
                y[i] += AS[j] * x[JA[j]];
            }
        }
    }
}

void compute_thread_row_partition(int M, int nz, int *num_threads, int *IRP, int **start_row, int **end_row) {
    *start_row = malloc((size_t)*num_threads * sizeof(int));
    *end_row = malloc((size_t)*num_threads * sizeof(int));
    int *nnz_per_thread_count = malloc((size_t)(*num_threads) * sizeof(int)); // Array per conteggio dei non zeri

    if (*start_row == NULL || *end_row == NULL || nnz_per_thread_count == NULL) {
        printf("Errore nell'allocazione della memoria per la partizione delle righe\n");
        exit(EXIT_FAILURE);
    }

    // Inizializza i valori
    for (int t = 0; t < *num_threads; t++) {
        (*start_row)[t] = -1;
        (*end_row)[t] = -1;
        nnz_per_thread_count[t] = 0; // Inizializza il conteggio a 0
    }

    int ideal_nnz_per_thread = nz / *num_threads; // Quota ideale di non zeri per ogni thread
    int current_thread = 0;
    int current_nnz = 0;

    // Assegna righe in modo da bilanciare i non zeri
    for (int i = 0; i < M; i++) {
        int nnz_in_row = IRP[i + 1] - IRP[i]; // Non zeri nella riga corrente

        if ((*start_row)[current_thread] == -1) {
            (*start_row)[current_thread] = i; // Imposta l'inizio per il thread corrente
        }

        current_nnz += nnz_in_row;
        nnz_per_thread_count[current_thread] += nnz_in_row;

        // Se il limite di non zeri per il thread è raggiunto, passa al prossimo thread
        if (current_nnz >= ideal_nnz_per_thread && current_thread < *num_threads - 1) {
            (*end_row)[current_thread] = i + 1; // Fine inclusiva per il thread corrente
            current_thread++;
            current_nnz = 0;
        }
    }

    // Assegna la fine per l'ultimo thread
    if (current_thread < *num_threads) {
        (*end_row)[current_thread] = M;
    }

    int valid_threads = 0;
    // Rimuovi i thread non validi (ad esempio, se ci sono più thread rispetto alle righe o i non zeri)
    for (int t = 0; t < *num_threads; t++) {
        if ((*start_row)[t] != -1 && (*end_row)[t] != -1 && nnz_per_thread_count[t] > 0) {
            // Mantieni il thread valido
            (*start_row)[valid_threads] = (*start_row)[t];
            (*end_row)[valid_threads] = (*end_row)[t];
            nnz_per_thread_count[valid_threads] = nnz_per_thread_count[t];
            valid_threads++;
        }
    }
    // Aggiorna il numero finale di thread
    *num_threads = valid_threads;

    free(nnz_per_thread_count);
}

struct matrixPerformance parallel_csr(struct matrixData *matrix_data, double *x, int num_threads) {
    int *IRP, *JA;
    double *AS;

    // Vettore di output del risultato y <- Ax
    double *y = malloc((size_t)matrix_data->M * sizeof(double));
    if (y == NULL) {
        printf("Errore nell'allocazione della memoria per il vettore di output y\n");
        exit(EXIT_FAILURE);
    }

    struct matrixPerformance node;
    node.seconds = 0.0;
    node.flops = 0.0;
    node.gigaFlops = 0.0;

    convert_to_csr(matrix_data->M, matrix_data->nz, matrix_data->row_indices, matrix_data->col_indices, matrix_data->values, &IRP, &JA, &AS);

    int *start_row, *end_row;

    compute_thread_row_partition(matrix_data->M, matrix_data->nz, &num_threads, IRP, &start_row, &end_row);

    double start = omp_get_wtime();
    matvec_csr_openMP(IRP, JA, AS, x, y, start_row, end_row, num_threads, matrix_data->nz, matrix_data->M);
    double end = omp_get_wtime();

    node.seconds = end - start;
    //node.relativeError = checkDifferencesOpenMP(y ,matrix_data->M);
    node.flops = 2.0 * matrix_data->nz;
    node.gigaFlops = node.flops / node.seconds / 1e9;

    free(start_row);
    free(end_row);
    free(y);
    free(IRP);
    free(JA);
    free(AS);

    return node;
}