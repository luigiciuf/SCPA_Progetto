#include <cstdio>
#include <cstdlib>
#include <chrono> 

#include <cmath>
#include "../CUDA_libs/csr_utils.h"
#include "../libs/data_structure.h"
#include "../CUDA_libs/csr_Operation.h"



double *y_CPU = nullptr;

matrixPerformance serial_csr_cuda(matrixData *matrix_data_host, double *x_h) {
    int *IRP, *JA;
    double *AS;

    int M = matrix_data_host->M;

    auto *y_h = static_cast<double *>(malloc(M * sizeof(double)));
    if (y_h == nullptr) {
        printf("Errore nell'allocazione della memoria per il vettore di output y\n");
        exit(EXIT_FAILURE);
    }

    convert_to_csr(M, matrix_data_host->nz, matrix_data_host->row_indices,
                   matrix_data_host->col_indices, matrix_data_host->values,
                   &IRP, &JA, &AS);

    // ⏱️ Misura il tempo con chrono
    auto start = std::chrono::high_resolution_clock::now();

    matvec_csr(M, IRP, JA, AS, x_h, y_h);

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> duration = end - start;

    // Salva il risultato per verifiche (se serve)
    y_CPU = static_cast<double *>(malloc(M * sizeof(double)));
    memcpy(y_CPU, y_h, M * sizeof(double));

    matrixPerformance node{};
    node.seconds = duration.count();
    node.flops = 2.0 * matrix_data_host->nz;
    node.gigaFlops = node.flops / node.seconds / 1e9;

    free(y_h);
    free(IRP);
    free(JA);
    free(AS);

    return node;
}