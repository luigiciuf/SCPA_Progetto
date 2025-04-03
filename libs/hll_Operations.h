#ifndef HLLOPERATIONS_H
#define HLLOPERATIONS_H
#include "../libs/data_structure.h"

struct matrixPerformance serial_hll(struct matrixData *matrix_data, double *x_h, int num_threads);
struct matrixPerformance parallel_hll(struct matrixData *matrix_data, double *x_h, int num_threads);

#endif //HLLOPERATIONS_H