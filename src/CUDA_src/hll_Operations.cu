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
matrixPerformance parallel_hll_cuda(matrixData *matrix_data_host, double *x_h) {
    double *d_x = nullptr, *d_y = nullptr;
    int M = matrix_data_host->M;

    cudaDeviceReset();

    // === Alloca host vector y ===
    double *y_h = (double *)malloc(M * sizeof(double));
    if (!y_h) {
        fprintf(stderr, "Errore allocazione vettore y\n");
        exit(EXIT_FAILURE);
    }
    memset(y_h, 0, M * sizeof(double));

    // === Costruzione matrice HLL standard su host ===
    HLL_Matrix *hllMatrixHost = (HLL_Matrix *)malloc(sizeof(HLL_Matrix));
    hllMatrixHost->num_blocks = (M + HackSize - 1) / HackSize;
    hllMatrixHost->blocks = (ELLPACK_Block *)malloc(hllMatrixHost->num_blocks * sizeof(ELLPACK_Block));
    if (!hllMatrixHost->blocks) {
        fprintf(stderr, "Errore allocazione blocchi ELLPACK\n");
        free(hllMatrixHost);
        exit(EXIT_FAILURE);
    }

    convert_to_hll_cuda(matrix_data_host, hllMatrixHost);

    // === Conversione in struttura "flattened" per GPU ===
    FlattenedHLLMatrix d_flat = build_flattened_hll_matrix(hllMatrixHost);

    // === Allocazione e copia di x su device ===
    cudaMalloc(&d_x, M * sizeof(double));
    cudaMemcpy(d_x, x_h, M * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_y, M * sizeof(double));
    cudaMemset(d_y, 0, M * sizeof(double));

    // === Configurazione griglia CUDA ===
    int block_size = 256;
    int grid_size = (M + block_size - 1) / block_size;

    dim3 BLOCK_DIM(block_size);
    dim3 GRID_DIM(grid_size);

    // === Timing ===
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // === Kernel launch ===
    matvec_hll_flat_cuda<<<GRID_DIM, BLOCK_DIM>>>(d_flat, d_x, d_y, M);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // === Copia dei risultati su host ===
    cudaMemcpy(y_h, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);

    // === Cleanup GPU ===
    cudaFree(d_flat.d_JA);
    cudaFree(d_flat.d_AS);
    cudaFree(d_flat.d_block_offsets);
    cudaFree(d_flat.d_block_max_nz);
    cudaFree(d_flat.d_block_rows);

    cudaFree(d_x);
    cudaFree(d_y);

    // === Cleanup Host ===
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        free(hllMatrixHost->blocks[i].JA);
        free(hllMatrixHost->blocks[i].AS);
    }
    free(hllMatrixHost->blocks);
    free(hllMatrixHost);
    free(y_h);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // === Performance ===
    matrixPerformance perf{};
    perf.seconds = milliseconds / 1000.0;
    perf.flops = 2.0 * matrix_data_host->nz;
    perf.gigaFlops = perf.flops / perf.seconds / 1e9;

    return perf;
}
