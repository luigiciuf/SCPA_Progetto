#include "../libs/data_structure.h"
#include "../libs/hll_ellpack_utils.h"
#include <tgmath.h>
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include <string.h>
#include "../libs/costants.h"


/* Funzione per calcolare il massimo numero di nonzeri per ciascuna riga */
void calculate_max_nz_in_row_in_block(const struct matrixData *matrix_data, int *nz_per_row) {
    for (int i = 0; i < matrix_data->nz; i++) {
        int row_idx = matrix_data->row_indices[i];
        nz_per_row[row_idx]++;
    }
}

/* Funzione per trovare il massimo numero di nonzeri all'interno di un intervallo di righe */
int find_max_nz(const int *nz_per_row, int start_row, int end_row) {
    int max_nz = 0;
    for (int i = start_row; i < end_row; i++) {
        if (nz_per_row[i] > max_nz)
            max_nz = nz_per_row[i];
    }
    return max_nz;
}

int find_max_nz_per_block(const int *nz_per_row, int start_row, int end_row) {
    int tot_nz = 0;
    for (int i = start_row; i < end_row; i++) {
        tot_nz += nz_per_row[i];
    }
    return tot_nz;
}

void convert_to_hll(struct matrixData *matrix_data, HLL_Matrix *hll_matrix) {
    int *row_start = calloc(matrix_data->M + 1, sizeof(int));
    if (!row_start) {
        fprintf(stderr, "Errore: allocazione memoria fallita per row_start.\n");
        exit(EXIT_FAILURE);
    }

    // Conta gli elementi in ogni riga
    for (int i = 0; i < matrix_data->nz; i++) {
        row_start[matrix_data->row_indices[i] + 1]++;
    }

    // Calcola gli offset cumulativi
    for (int i = 1; i <= matrix_data->M; i++) {
        row_start[i] += row_start[i - 1];
    }

    int *sorted_col_indices = malloc(matrix_data->nz * sizeof(int));
    double *sorted_values = malloc(matrix_data->nz * sizeof(double));
    if (!sorted_col_indices || !sorted_values) {
        fprintf(stderr, "Errore: allocazione memoria fallita per array ordinati.\n");
        free(row_start);
        exit(EXIT_FAILURE);
    }

    // Ordina i dati per riga
    for (int i = 0; i < matrix_data->nz; i++) {
        int row = matrix_data->row_indices[i];
        int pos = row_start[row]++;
        sorted_col_indices[pos] = matrix_data->col_indices[i];
        sorted_values[pos] = matrix_data->values[i];
    }

    // Ripristina row_start
    for (int i = matrix_data->M; i > 0; i--) {
        row_start[i] = row_start[i - 1];
    }
    row_start[0] = 0;

    int *nz_per_row = calloc(matrix_data->M, sizeof(int));
    if (!nz_per_row) {
        fprintf(stderr, "Errore: Allocazione fallita per nz_per_row.\n");
        exit(EXIT_FAILURE);
    }

    calculate_max_nz_in_row_in_block(matrix_data, nz_per_row);

    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        int start_row = block_idx * HackSize;
        int end_row = (block_idx + 1) * HackSize;
        if (end_row > matrix_data->M) end_row = matrix_data->M;

        hll_matrix->blocks[block_idx].nz_per_block = find_max_nz_per_block(nz_per_row, start_row, end_row);
        hll_matrix->blocks[block_idx].max_nz_per_row = find_max_nz(nz_per_row, start_row, end_row);

        int max_nz_per_row = hll_matrix->blocks[block_idx].max_nz_per_row;
        int rows_in_block = end_row - start_row;
        int size_of_arrays = max_nz_per_row * rows_in_block;
        if (max_nz_per_row < 0 || rows_in_block < 0) {
            fprintf(stderr, "Errore: Valori invalidi per il blocco %d: %d - %d\n", block_idx, rows_in_block, max_nz_per_row);
            exit(EXIT_FAILURE);
        }

        hll_matrix->blocks[block_idx].JA = calloc(size_of_arrays, sizeof(int));
        hll_matrix->blocks[block_idx].AS = calloc(size_of_arrays, sizeof(double));
        if (!hll_matrix->blocks[block_idx].JA || !hll_matrix->blocks[block_idx].AS) {
            fprintf(stderr, "Errore: allocazione memoria fallita per il blocco %d.\n", block_idx);
            for (int k = 0; k < block_idx; k++) {
                free(hll_matrix->blocks[k].JA);
                free(hll_matrix->blocks[k].AS);
            }
            free(nz_per_row);
            exit(EXIT_FAILURE);
        }

        memset(hll_matrix->blocks[block_idx].JA, -1, size_of_arrays * sizeof(int));
        memset(hll_matrix->blocks[block_idx].AS, 0, size_of_arrays * sizeof(double));

        for (int i = start_row; i < end_row; i++) {
            int row_offset = (i - start_row) * max_nz_per_row;
            int row_nz_start = row_start[i];
            int row_nz_end = row_start[i + 1];

            int pos = 0;
            int last_col_idx = -1;

            // Assegna i valori nel formato HLL
            for (int j = row_nz_start; j < row_nz_end; j++) {
                if (pos >= max_nz_per_row) break;
                int index = row_offset + pos;
                hll_matrix->blocks[block_idx].JA[index] = sorted_col_indices[j];
                hll_matrix->blocks[block_idx].AS[index] = sorted_values[j];
                last_col_idx = sorted_col_indices[j];
                pos++;
            }

            // Aggiunta del padding
            while (pos < max_nz_per_row) {
                int index = row_offset + pos;
                hll_matrix->blocks[block_idx].JA[index] = last_col_idx;
                hll_matrix->blocks[block_idx].AS[index] = 0.0;
                pos++;
            }
        }
    }

    free(row_start);
    free(sorted_col_indices);
    free(sorted_values);
    free(nz_per_row);
}

void matvec_Hll_serial(const HLL_Matrix *hll_matrix, const double *x, double *y, int max_row_in_matrix) {
    for (int blockID = 0; blockID < hll_matrix->num_blocks; blockID++) {
        /* Calcolo delle righe di inizio e fine del blocco */
        int start_row = blockID * HackSize;
        int end_row = (blockID + 1) * HackSize;
        if (end_row > max_row_in_matrix) end_row = max_row_in_matrix;
        int row_offset = 0;
        /* Scorrimento delle righe di un unico blocco */
        for (int i = start_row; i < end_row; i++) {
            y[i] = 0.0;
            /* Scorrimento della riga selezionata (sarà lunga maxnz) */
            for (int j = 0; j < hll_matrix->blocks[blockID].max_nz_per_row; j++) {
                y[i] += hll_matrix->blocks[blockID].AS[j + row_offset] * x[hll_matrix->blocks[blockID].JA[j + row_offset]];
            }
            /* Incremento dell'offset per passare alla riga successiva */
            row_offset += hll_matrix->blocks[blockID].max_nz_per_row;
        }
    }
}

void matvec_Hll(const HLL_Matrix *hll_matrix, const double *x, double *y, int num_threads, const int *start_block, const int *end_block, int max_row_in_matrix) {
    #pragma omp parallel num_threads(num_threads)
    {
        int tid = omp_get_thread_num();
        /* Scorrimento dei blocchi assegnati al thread tid */
        for (int block_id = start_block[tid]; block_id <= end_block[tid]; block_id++) {
            /* Calcolo delle righe di inizio e fine del blocco */
            int start_row = block_id * HackSize;
            int end_row = (block_id + 1) * HackSize;
            if (end_row > max_row_in_matrix) end_row = max_row_in_matrix;

            int row_offset = 0;
            /* Scorrimento delle righe di un unico blocco */
            for (int i = start_row; i < end_row; i++) {
                y[i] = 0.0;
                /* Scorrimento della riga selezionata (sarà lunga maxnz) */
                for (int j = 0; j < hll_matrix->blocks[block_id].max_nz_per_row; j++) {
                    y[i] += hll_matrix->blocks[block_id].AS[j + row_offset] * x[hll_matrix->blocks[block_id].JA[j + row_offset]];
                }
                /* Incremento dell'offset per passare alla riga successiva */
                row_offset += hll_matrix->blocks[block_id].max_nz_per_row;
            }
        }
    }
}