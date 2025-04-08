#include "../../libs/costants.h"
#include "../CUDA_libs/hll_ellpack_utils.h"
#include "../../libs/data_structure.h"
#include "../CUDA_libs/csr_Operation.h"

#include "../CUDA_libs/costants.h"


//implementazione cuda seriale hll

matrixPerformance serial_hll_cuda(matrixData *matrix_data_host, double *x_h) {
    double *d_x = nullptr;
    double *d_y = nullptr;
    int M = matrix_data_host->M;

    cudaDeviceReset();

    HLL_Matrix *hllMatrixHost = static_cast<HLL_Matrix *>(malloc(sizeof(HLL_Matrix)));
    if (!hllMatrixHost) {
        fprintf(stderr, "Errore: Allocazione fallita per HLL_Matrix.\n");
        exit(EXIT_FAILURE);
    }

    hllMatrixHost->num_blocks = (M + HackSize - 1) / HackSize;
    hllMatrixHost->blocks = static_cast<ELLPACK_Block *>(malloc(hllMatrixHost->num_blocks * sizeof(ELLPACK_Block)));
    if (!hllMatrixHost->blocks) {
        fprintf(stderr, "Errore: Allocazione fallita per i blocchi ELLPACK.\n");
        free(hllMatrixHost);
        exit(EXIT_FAILURE);
    }

    auto *y_h = static_cast<double *>(malloc(M * sizeof(double)));
    if (!y_h) {
        fprintf(stderr, "Errore: Allocazione fallita per il vettore y_h.\n");
        free(hllMatrixHost->blocks);
        free(hllMatrixHost);
        exit(EXIT_FAILURE);
    }
    convert_to_hll_cuda(matrix_data_host, hllMatrixHost);
    // Conversione in HLL


    // Timer con cudaEvent
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    matvec_Hll_serial_CUDA(hllMatrixHost, x_h, y_h, M);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    matrixPerformance node{};
    node.seconds = milliseconds / 1000.0;
    node.flops = 2.0 * matrix_data_host->nz;
    node.gigaFlops = node.flops / node.seconds / 1e9;
    // Cleanup
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        cudaFree(hllMatrixHost->blocks[i].JA);
        cudaFree(hllMatrixHost->blocks[i].AS);
    }

    cudaFree(d_x);
    cudaFree(d_y);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    free(y_h);
    free(hllMatrixHost->blocks);
    free(hllMatrixHost);

    return node;
}