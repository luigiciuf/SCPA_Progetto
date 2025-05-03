#ifndef HLLTOOL_H
#define HLLTOOL_H
#include "../../libs/data_structure.h"

void convert_to_hll_cuda(matrixData *matrix_data, HLL_Matrix *hll_matrix);

int find_max_nz_per_block(const int *nz_per_row, int start_row, int end_row) ;
int find_max_nz(const int *nz_per_row, int start_row, int end_row);
void calculate_max_nz_in_row_in_block(const matrixData *matrix_data, int *nz_per_row);
void matvec_Hll_serial_CUDA(const HLL_Matrix *hll_matrix, const double *x, double *y, int max_row_in_matrix);
__global__ void matvec_Hll_cuda_SH(const HLL_Matrix *d_hll_matrix, const double *d_x, double *d_y, int M);
FlattenedHLLMatrix build_flattened_hll_matrix(const HLL_Matrix *hll_matrix);
__global__ void matvec_hll_flat_cuda(
    const FlattenedHLLMatrix hll,
    const double* __restrict__ x,
    double* __restrict__ y,
    int M);
#endif //HLLTOOL_H