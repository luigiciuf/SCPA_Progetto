#include <stdlib.h>
#include <stdio.h>
#include <omp.h>
#include "csr_utils.h"
/* Funzione per convertire la matrice in formato CSR */
void convert_to_csr(int M, int nz, const int *row_indices, const int *col_indices, const double *values, int **IRP, int **JA, double **AS) {
    *IRP = (int *)malloc((M + 1) * sizeof(int));    // Dimensione del vettore M
    *JA = (int *)malloc(nz * sizeof(int));          // Dimensione del vettore NZ - 1
    *AS = (double *)malloc(nz * sizeof(double));    // Dimensione del vettore NZ - 1

    if (*IRP == NULL || *JA == NULL || *AS == NULL) {
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

    int *row_position = malloc(M * sizeof(int));
    if (row_position == NULL) {
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