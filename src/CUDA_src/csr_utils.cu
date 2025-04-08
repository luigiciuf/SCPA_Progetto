#include <cstdlib>
#include <cstdio>

#include "../CUDA_libs/csr_utils.h"
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