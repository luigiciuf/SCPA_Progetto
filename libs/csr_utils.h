#ifndef CSR_H
#define CSR_H

/* Funzione per convertire la matrice */
void convert_to_csr(int M, int nz, const int *row_indices, const int *col_indices, const double *values, int **IRP, int **JA, double **AS);


#endif