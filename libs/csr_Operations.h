#ifndef CSRSERIALIZED_H
#define CSRSERIALIZED_H
#include "../libs/data_structure.h"

/* Calcolo serializzato */
struct matrixPerformance serial_csr(struct matrixData *matrix_data, double *x, int num_threads);
void matvec_csr(int M, const int *IRP, const int *JA, const double *AS, const double *x, double *y);

#endif