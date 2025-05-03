#include <cstdlib>
#include <cstdio>

#include "../CUDA_libs/csr_utils.h"
#include "../CUDA_libs/costants.h"

/* Funzione per convertire la matrice in formato CSR */
void convert_to_csr(int M, int nz, const int *row_indices, const int *col_indices, const double *values, int **IRP, int **JA, double **AS) {
    *IRP = static_cast<int *>(malloc((M + 1) * sizeof(int)));    // Dimensione del vettore M
    *JA = static_cast<int *>(malloc(nz * sizeof(int)));          // Dimensione del vettore NZ - 1
    *AS = static_cast<double *>(malloc(nz * sizeof(double)));    // Dimensione del vettore NZ - 1

    if (*IRP == nullptr || *JA == nullptr || *AS == nullptr) {
        fprintf(stderr, "Errore nell'allocazione della memoria.\n");
        exit(1);
    }

    for (int i = 0; i <= M; i++) {
        (*IRP)[i] = 0;
    }

    for (int i = 0; i < nz; i++) {
        if (row_indices[i] < 0 || row_indices[i] >= M) {
            fprintf(stderr, "Errore: l'indice di riga è fuori dai limiti.\n");
            exit(1);
        }
        (*IRP)[row_indices[i] + 1]++;
    }

    for (int i = 0; i < M; i++) {
        (*IRP)[i + 1] += (*IRP)[i];
    }
    if (M <= 0) {
        fprintf(stderr, "Errore: valore di M non valido (%d)\n", M);
        exit(EXIT_FAILURE);
    }

    int *row_position = static_cast<int *>(malloc(M * sizeof(int)));
    if (row_position == nullptr) {
        fprintf(stderr, "Errore nell'allocazione di row_position.\n");
        exit(1);
    }

    for (int i = 0; i < M; i++) {
        row_position[i] = 0;
    }

    for (int i = 0; i < nz; i++) {
        int row = row_indices[i];
        int pos = (*IRP)[row] + row_position[row];
        (*JA)[pos] = col_indices[i];
        (*AS)[pos] = values[i];
        row_position[row]++;
    }

    free(row_position);
}
/* Prodotto matrice-vettore serializzato su CPU */
void matvec_csr(int M, const int *IRP, const int *JA, const double *AS, double *x, double *y) {
    for (int i = 0; i < M; i++) {
        y[i] = 0.0;
        for (int j = IRP[i]; j < IRP[i + 1]; j++) {
            y[i] += AS[j] * x[JA[j]];
        }
    }
}


/* kernel row based*/
__global__ void gpuMatVec_csr(const int *d_IRP, const int *d_JA, const double *d_AS, const double *d_x, double *d_y, int M) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;

    /* Controllo che la riga non ecceda il numero totale di righe */
    if (row >= M) return;

    double sum = 0.0;
    /* Ciascun thread si fa carico di una riga */
    for (int index = d_IRP[row]; index < d_IRP[row + 1]; index++)
        sum += d_AS[index] * d_x[d_JA[index]];
    d_y[row] = sum;
}

/* kernel warp based */

__global__ void gpuMatVec_csr_warp(const int *d_IRP, const int *d_JA,const double *d_AS, const double *d_x,double *d_y, int M) {

    int thread_id = blockIdx.x * blockDim.x + threadIdx.x;
    int warp_id = thread_id / WARPSIZE;
    int lane_id = threadIdx.x & (WARPSIZE-1);

    if (warp_id >= M) return;

    // I primi due thread del warp leggono gli offset della riga
    int row_start = 0, row_end = 0;
    if (lane_id == 0) row_start = d_IRP[warp_id];
    if (lane_id == 1) row_end = d_IRP[warp_id + 1];

    row_start = __shfl_sync(0xFFFFFFFF, row_start, 0);
    row_end   = __shfl_sync(0xFFFFFFFF, row_end, 1);

    double sum = 0.0;

    for (int i = row_start + lane_id; i < row_end; i += 32) {
    sum += d_AS[i] * d_x[d_JA[i]];
    }

    // Somma dei risultati all'interno del warp
    for (int offset = WARPSIZE; offset > 0; offset /= 2)
    sum += __shfl_down_sync(0xFFFFFFFF, sum, offset);

    if (lane_id == 0)
    d_y[warp_id] = sum;

    }