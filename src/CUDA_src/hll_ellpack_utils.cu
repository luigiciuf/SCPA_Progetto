#include <cstdlib>
#include <cstdio>
#include <cstring>

#include "../../libs/data_structure.h"
#include "../../libs/hll_ellpack_utils.h"
#include "../../libs/costants.h"
#include "../CUDA_libs/hll_ellpack_utils.h"

/* Funzione per calcolare il massimo numero di nonzeri per ciascuna riga */
void calculate_max_nz_in_row_in_block(const matrixData *matrix_data, int *nz_per_row) {
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

// ============================
// 2. convert_to_hll_cuda MIGLIORATO
// ============================
void convert_to_hll_cuda(matrixData *matrix_data, HLL_Matrix *hll_matrix) {
    int *row_start = (int *)calloc(matrix_data->M + 1, sizeof(int));
    if (!row_start) exit(EXIT_FAILURE);

    for (int i = 0; i < matrix_data->nz; i++) {
        row_start[matrix_data->row_indices[i] + 1]++;
    }
    for (int i = 1; i <= matrix_data->M; i++) {
        row_start[i] += row_start[i - 1];
    }

    int *sorted_col_indices = (int *)malloc(matrix_data->nz * sizeof(int));
    double *sorted_values = (double *)malloc(matrix_data->nz * sizeof(double));
    if (!sorted_col_indices || !sorted_values) exit(EXIT_FAILURE);

    for (int i = 0; i < matrix_data->nz; i++) {
        int row = matrix_data->row_indices[i];
        int pos = row_start[row]++;
        sorted_col_indices[pos] = matrix_data->col_indices[i];
        sorted_values[pos] = matrix_data->values[i];
    }
    for (int i = matrix_data->M; i > 0; i--) row_start[i] = row_start[i - 1];
    row_start[0] = 0;

    int *nz_per_row = (int *)calloc(matrix_data->M, sizeof(int));
    calculate_max_nz_in_row_in_block(matrix_data, nz_per_row);

    for (int block_idx = 0; block_idx < hll_matrix->num_blocks; block_idx++) {
        int start_row = block_idx * HackSize;
        int end_row = (block_idx + 1) * HackSize;
        if (end_row > matrix_data->M) end_row = matrix_data->M;

        hll_matrix->blocks[block_idx].nz_per_block = find_max_nz_per_block(nz_per_row, start_row, end_row);
        hll_matrix->blocks[block_idx].max_nz_per_row = find_max_nz(nz_per_row, start_row, end_row);

        int max_nz_per_row = hll_matrix->blocks[block_idx].max_nz_per_row;
        int rows_in_block = end_row - start_row;

        if (max_nz_per_row <= 0) {
            max_nz_per_row = 1;
            hll_matrix->blocks[block_idx].max_nz_per_row = 1;
            hll_matrix->blocks[block_idx].size_of_arrays = rows_in_block;
            hll_matrix->blocks[block_idx].JA = (int *)calloc(rows_in_block, sizeof(int));
            hll_matrix->blocks[block_idx].AS = (double *)calloc(rows_in_block, sizeof(double));
            continue;
        }

        int size_of_arrays = max_nz_per_row * rows_in_block;
        hll_matrix->blocks[block_idx].size_of_arrays = size_of_arrays;

        hll_matrix->blocks[block_idx].JA = (int *)calloc(size_of_arrays, sizeof(int));
        hll_matrix->blocks[block_idx].AS = (double *)calloc(size_of_arrays, sizeof(double));

        for (int i = start_row; i < end_row; i++) {
            int row_offset = (i - start_row) * max_nz_per_row;
            int row_nz_start = row_start[i];
            int row_nz_end = row_start[i + 1];

            int pos = 0;
            int last_col_idx = 0;

            for (int j = row_nz_start; j < row_nz_end && pos < max_nz_per_row; j++, pos++) {
                int index = row_offset + pos;
                hll_matrix->blocks[block_idx].JA[index] = sorted_col_indices[j];
                hll_matrix->blocks[block_idx].AS[index] = sorted_values[j];
                last_col_idx = sorted_col_indices[j];
            }
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

void matvec_Hll_serial_CUDA(const HLL_Matrix *hll_matrix, const double *x, double *y, int max_row_in_matrix) {
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
         // 🧹 Libera il blocco appena usato
         free(hll_matrix->blocks[blockID].JA);
         free(hll_matrix->blocks[blockID].AS);
         ((HLL_Matrix *)hll_matrix)->blocks[blockID].JA = NULL;
         ((HLL_Matrix *)hll_matrix)->blocks[blockID].AS = NULL;
    }

}
/* Prodotto matrice-vettore parallelo su GPU - ciascun thread di un blocco prende in carico una riga */
__global__ void matvec_Hll_cuda_SH(const HLL_Matrix *d_hll_matrix, const double *d_x, double *d_y, int M) {
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;

    if (global_row >= M) return;

    // Identifico il blocco a livello globale
    int block_id = global_row / HackSize;
    // Identifico la riga all'interno del blocco
    int local_row = global_row % HackSize;

    const ELLPACK_Block *block = &d_hll_matrix->blocks[block_id];

    // Individuo la riga da cui devo partire ad effettuare il prodotto
    int row_offset = local_row * block->max_nz_per_row;

    // Calcola il prodotto matrice-vettore
    double sum = 0.0;
    for (int j = 0; j < block->max_nz_per_row; j++)
        sum += block->AS[row_offset + j] * d_x[block->JA[row_offset + j]];

    d_y[global_row] = sum;
}

FlattenedHLLMatrix build_flattened_hll_matrix(const HLL_Matrix *hll_matrix) {
    int num_blocks = hll_matrix->num_blocks;

    int total_nnz = 0;
    for (int i = 0; i < num_blocks; i++) {
        total_nnz += hll_matrix->blocks[i].size_of_arrays;
    }

    int* h_JA = (int*)malloc(total_nnz * sizeof(int));
    double* h_AS = (double*)malloc(total_nnz * sizeof(double));
    int* h_block_offsets = (int*)malloc(num_blocks * sizeof(int));
    int* h_block_max_nz = (int*)malloc(num_blocks * sizeof(int));
    int* h_block_rows = (int*)malloc(num_blocks * sizeof(int));

    if (!h_JA || !h_AS || !h_block_offsets || !h_block_max_nz || !h_block_rows) {
        fprintf(stderr, "Errore: allocazione memoria host in flattening HLL\n");
        exit(EXIT_FAILURE);
    }

    int offset = 0;
    for (int b = 0; b < num_blocks; b++) {
        const ELLPACK_Block *block = &hll_matrix->blocks[b];

        int max_nz = block->max_nz_per_row;
        if (max_nz <= 0) {
            fprintf(stderr, "Errore: max_nz_per_row = %d nel blocco %d\n", max_nz, b);
            exit(EXIT_FAILURE);
        }

        int block_size = block->size_of_arrays;
        int block_rows = block_size / max_nz;

        h_block_offsets[b] = offset;
        h_block_max_nz[b] = max_nz;
        h_block_rows[b] = block_rows;

        for (int i = 0; i < block_size; i++) {
            h_JA[offset + i] = block->JA[i];
            h_AS[offset + i] = block->AS[i];
        }

        offset += block_size;
    }

    FlattenedHLLMatrix d_matrix;
    d_matrix.num_blocks = num_blocks;
    d_matrix.total_nnz = total_nnz;

    cudaMalloc(&d_matrix.d_JA, total_nnz * sizeof(int));
    cudaMalloc(&d_matrix.d_AS, total_nnz * sizeof(double));
    cudaMalloc(&d_matrix.d_block_offsets, num_blocks * sizeof(int));
    cudaMalloc(&d_matrix.d_block_max_nz, num_blocks * sizeof(int));
    cudaMalloc(&d_matrix.d_block_rows, num_blocks * sizeof(int));

    cudaMemcpy(d_matrix.d_JA, h_JA, total_nnz * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix.d_AS, h_AS, total_nnz * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix.d_block_offsets, h_block_offsets, num_blocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix.d_block_max_nz, h_block_max_nz, num_blocks * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_matrix.d_block_rows, h_block_rows, num_blocks * sizeof(int), cudaMemcpyHostToDevice);

    free(h_JA);
    free(h_AS);
    free(h_block_offsets);
    free(h_block_max_nz);
    free(h_block_rows);

    return d_matrix;
}


__global__ void matvec_hll_flat_cuda(const FlattenedHLLMatrix hll,const double* __restrict__ x,double* __restrict__ y,int M){
    int global_row = blockIdx.x * blockDim.x + threadIdx.x;
    if (global_row >= M) return;

    int block_id = global_row / HackSize;
    int local_row = global_row % HackSize;

    int offset      = hll.d_block_offsets[block_id];
    int max_nz      = hll.d_block_max_nz[block_id];
    int block_rows  = hll.d_block_rows[block_id];

    if (local_row >= block_rows) {
        y[global_row] = 0.0;
        return;
    }

    double sum = 0.0;
    for (int j = 0; j < max_nz; j++) {
        int idx = offset + j * block_rows + local_row;
        int col = hll.d_JA[idx];
        double val = hll.d_AS[idx];
        if (col >= 0 && col < M) {
            sum += val * __ldg(&x[col]);
        }
    }

    y[global_row] = sum;
}