#ifndef CSRTOOL_H
#define CSRTOOL_H

/* Funzione per convertire la matrice */
void convert_to_csr(int M, int nz, const int *row_indices, const int *col_indices, const double *values, int **IRP, int **JA, double **AS);

void matvec_csr(int M, const int *IRP, const int *JA, const double *AS, double *x, double *y);

#endif //CSRTOOL_H