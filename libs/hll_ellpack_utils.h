#ifndef HLL_ELLPACK_TOOL_H
#define HLL_ELLPACK_TOOL_H

#include "../libs/data_structure.h"


void convert_to_hll(struct matrixData *matrix_data, HLL_Matrix *hll_matrix);

void matvec_Hll(const HLL_Matrix *hll_matrix, const double *x, double *y, int num_threads, const int *start_block, const int *end_block, int max_row_in_matrix); ;

void calculate_max_nz_in_row_in_block(const struct matrixData *matrix_data, int *nz_per_row);

void matvec_Hll_serial(const HLL_Matrix *hll_matrix, const double *x, double *y, int max_row_in_matrix);

struct matrixPerformance serial_hll(struct matrixData *matrix_data, double *x_h, int num_threads);

#endif // HLL_ELLPACK_TOOL_H