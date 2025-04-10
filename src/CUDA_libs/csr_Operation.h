#ifndef CSROPERATIONS_H
#define CSROPERATIONS_H

#include "../libs/data_structure.h"

matrixPerformance serial_csr_cuda(matrixData *matrix_data_host, double *x_h);

matrixPerformance parallel_csr_cuda(matrixData *matrix_data_host, double *x_h);

#endif //CSROPERATIONS_H