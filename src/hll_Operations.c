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

void distribute_blocks_to_threads(struct matrixData *matrix_data, HLL_Matrix *hll_matrix, int num_threads, int **start_block, int **end_block, int *valid_threads) {
    *start_block = (int *)malloc((size_t)num_threads * sizeof(int));
    *end_block = (int *)malloc((size_t)num_threads * sizeof(int));
    if (!*start_block || !*end_block) {
        fprintf(stderr, "Errore: Allocazione fallita per start_block o end_block.\n");
        exit(EXIT_FAILURE);
    }

    int *non_zero_per_row = calloc(matrix_data->M, sizeof(int));
    if (!non_zero_per_row) {
        fprintf(stderr, "Errore: Allocazione fallita per non_zero_per_row.\n");
        exit(EXIT_FAILURE);
    }

    // Calcola il numero di non-zero per ogni riga
    calculate_max_nz_in_row_in_block(matrix_data, non_zero_per_row);

    // Calcola il numero totale di non-zero
    int total_non_zero = 0;
    for (int i = 0; i < matrix_data->M; i++) {
        total_non_zero += non_zero_per_row[i];
    }

    int non_zero_per_thread = total_non_zero / num_threads;
    int current_non_zero = 0;
    int thread_id = 0;

    (*start_block)[thread_id] = 0; // Primo blocco assegnato al primo thread

    /* Distribuzione dei blocchi ai vari thread */
    for (int block_id = 0; block_id < hll_matrix->num_blocks; block_id++) {
        // Aggiunge i non-zero del blocco corrente
        current_non_zero += hll_matrix->blocks[block_id].nz_per_block;

        // Controlla se è necessario passare al prossimo thread
        if (current_non_zero >= non_zero_per_thread && thread_id < num_threads - 1) {
            (*end_block)[thread_id] = block_id; // Assegna il blocco finale per il thread corrente

            // Verifica che il thread abbia un sottoinsieme non nullo
            if ((*end_block)[thread_id] >= (*start_block)[thread_id]) {
                //printf("HLL Thread %d: blocchi da %d a %d, non-zero = %d\n", thread_id, (*start_block)[thread_id], (*end_block)[thread_id], current_non_zero);
                thread_id++;
                (*start_block)[thread_id] = block_id + 1; // Inizia il prossimo thread dal blocco successivo
                current_non_zero = 0; // Reset per il prossimo thread
            } else {
                // Se il sottoinsieme di blocchi è nullo, riassigna al thread precedente
                (*end_block)[thread_id - 1] = block_id;
                current_non_zero += hll_matrix->blocks[block_id].nz_per_block;
            }
        }
    }

    if (hll_matrix->num_blocks > 1) {
        (*end_block)[thread_id] = hll_matrix->num_blocks - 1;

        // Verifica che il thread abbia un sottoinsieme non nullo
        if ((*end_block)[thread_id] >= (*start_block)[thread_id]) {
            //printf("HLL Thread %d: blocchi da %d a %d, non-zero = %d\n", thread_id, (*start_block)[thread_id], (*end_block)[thread_id], current_non_zero);
            *valid_threads = thread_id + 1;
        } else {
            (*end_block)[thread_id - 1] = (*end_block)[thread_id];
            *valid_threads = thread_id;
        }
    } else {
        *valid_threads = thread_id;
    }

    // Verifica che tutti i blocchi siano stati assegnati una sola volta e che il numero totale corrisponda
    int assigned_blocks = 0;
    int assigned_non_zero = 0;
    for (int i = 0; i < *valid_threads; i++) {
        if ((*start_block)[i] > (*end_block)[i]) {
            fprintf(stderr, "Errore: Assegnazione non valida per il thread %d.\n", i);
            exit(EXIT_FAILURE);
        }
        assigned_blocks += (*end_block)[i] - (*start_block)[i] + 1;
        for (int block_id = (*start_block)[i]; block_id <= (*end_block)[i]; block_id++) {
            assigned_non_zero += hll_matrix->blocks[block_id].nz_per_block;
        }
    }

    if (assigned_blocks != hll_matrix->num_blocks) {
        fprintf(stderr, "Errore: Il numero di blocchi assegnati (%d) non corrisponde al numero totale di blocchi (%d).\n", assigned_blocks, hll_matrix->num_blocks);
        exit(EXIT_FAILURE);
    }

    if (assigned_non_zero != total_non_zero) {
        fprintf(stderr, "Errore: Il numero totale di non-zero assegnati (%d) non corrisponde al totale nella matrice (%d).\n", assigned_non_zero, total_non_zero);
        exit(EXIT_FAILURE);
    }

    /*printf("HLL Numero totale di thread validi: %d\n", *valid_threads);
    printf("HLL Verifica completata: tutti i blocchi e i non-zero sono stati assegnati correttamente.\n");*/

    free(non_zero_per_row);
}
// Funzione principale per calcolare il prodotto parallelo
struct matrixPerformance parallel_hll(struct matrixData *matrix_data, double *x_h, int num_threads) {
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

    int *start_block = NULL;
    int *end_block = NULL;
    int valid_threads = 0;

    /* Distribuzione dei blocchi tra i thread */
    distribute_blocks_to_threads(matrix_data, hll_matrix, num_threads, &start_block, &end_block, &valid_threads);

    double *y = malloc((size_t)M * sizeof(double));
    if (!y) {
        fprintf(stderr, "Errore: Allocazione fallita per il vettore y.\n");
        free(hll_matrix->blocks);
        free(hll_matrix);
        exit(EXIT_FAILURE);
    }

    double start = omp_get_wtime();
    matvec_Hll(hll_matrix, x_h, y, valid_threads, start_block, end_block, matrix_data->M);
    double end = omp_get_wtime();

    const double time_spent = end - start;

    struct matrixPerformance performance;
    performance.seconds = time_spent;
    performance.flops = 2.0 * matrix_data->nz;
    performance.gigaFlops = performance.flops / performance.seconds / 1e9;

    // Libera memoria
    free(y);
    free(start_block);
    free(end_block);
    for (int i = 0; i < hll_matrix->num_blocks; i++) {
        free(hll_matrix->blocks[i].JA);
        free(hll_matrix->blocks[i].AS);
    }
    free(hll_matrix->blocks);
    free(hll_matrix);

    return performance;
}