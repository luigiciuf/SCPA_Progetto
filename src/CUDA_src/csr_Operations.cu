#include <cstdio>
#include <cstdlib>
#include <chrono> 
#include <cuda_runtime.h>

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
matrixPerformance parallel_csr_cuda(matrixData *matrix_data_host, double *x_h) {
    int M = matrix_data_host->M;
    int NZ = matrix_data_host->nz;

    // Allocazione host
    int *IRP, *JA;
    double *AS;
    convert_to_csr(M, NZ, matrix_data_host->row_indices, matrix_data_host->col_indices, matrix_data_host->values, &IRP, &JA, &AS);

    // Allocazione device
    int *d_IRP, *d_JA;
    double *d_AS, *d_x, *d_y;
    cudaMalloc((void**)&d_IRP, (M + 1) * sizeof(int));
    cudaMalloc((void**)&d_JA, NZ * sizeof(int));
    cudaMalloc((void**)&d_AS, NZ * sizeof(double));
    cudaMalloc((void**)&d_x, matrix_data_host->N * sizeof(double));
    cudaMalloc((void**)&d_y, M * sizeof(double));

    // Copia da host a device
    cudaMemcpy(d_IRP, IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_JA, JA, NZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AS, AS, NZ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x_h, matrix_data_host->N * sizeof(double), cudaMemcpyHostToDevice);

    // Setup kernel
    int threadsPerBlock = 128;
    int blocksPerGrid = (M + threadsPerBlock - 1) / threadsPerBlock;

    // Timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Lancio kernel
    gpuMatVec_csr<<<blocksPerGrid, threadsPerBlock>>>(d_IRP, d_JA, d_AS, d_x, d_y, M);
    
     // ===== Sincronizzazione e controllo errori =====
     cudaEventRecord(stop);
     cudaEventSynchronize(stop);
 
     cudaError_t err = cudaGetLastError();
     if (err != cudaSuccess) {
         fprintf(stderr, "[CUDA ERROR] %s\n", cudaGetErrorString(err));
     }
 
     float milliseconds = 0;
     cudaEventElapsedTime(&milliseconds, start, stop);
 
     double seconds = milliseconds / 1000.0;
     double flops = 2.0 * NZ;
     double gflops = flops / seconds / 1e9;
 
     // ===== Analisi e warning su risultati anomali =====
     if (seconds < 1e-6 || gflops > 10000.0) {
         printf("⚠️  [WARNING] Possibile anomalia: M=%d, NZ=%d, t=%.10f s, GFLOPS=%.2f\n", M, NZ, seconds, gflops);
     }
 
     // ===== Cleanup =====
     cudaFree(d_IRP);
     cudaFree(d_JA);
     cudaFree(d_AS);
     cudaFree(d_x);
     cudaFree(d_y);
 
     free(IRP);
     free(JA);
     free(AS);
 
     cudaEventDestroy(start);
     cudaEventDestroy(stop);
 
     // ===== Return performance =====
     matrixPerformance perf;
     perf.seconds = seconds;
     perf.flops = flops;
     perf.gigaFlops = gflops;
 
     return perf;
 }

 matrixPerformance parallel_csr_cuda_warp(matrixData *matrix_data_host, double *x_h) {
    int M = matrix_data_host->M;
    int NZ = matrix_data_host->nz;

    int *IRP, *JA;
    double *AS;
    convert_to_csr(M, NZ, matrix_data_host->row_indices, matrix_data_host->col_indices,
                   matrix_data_host->values, &IRP, &JA, &AS);

    int *d_IRP, *d_JA;
    double *d_AS, *d_x, *d_y;
    cudaMalloc(&d_IRP, (M + 1) * sizeof(int));
    cudaMalloc(&d_JA, NZ * sizeof(int));
    cudaMalloc(&d_AS, NZ * sizeof(double));
    cudaMalloc(&d_x, matrix_data_host->N * sizeof(double));
    cudaMalloc(&d_y, M * sizeof(double));

    cudaMemcpy(d_IRP, IRP, (M + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_JA, JA, NZ * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_AS, AS, NZ * sizeof(double), cudaMemcpyHostToDevice);
    cudaMemcpy(d_x, x_h, matrix_data_host->N * sizeof(double), cudaMemcpyHostToDevice);

    // Impostazioni kernel warp-based
    int threadsPerBlock = 128; // 32 * 32
    int warpsPerBlock = threadsPerBlock / 32;
    int numWarps = (M + 1 + warpsPerBlock - 1) / warpsPerBlock;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    gpuMatVec_csr_warp<<<numWarps, threadsPerBlock>>>(d_IRP, d_JA, d_AS, d_x, d_y, M);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess)
        fprintf(stderr, "[CUDA ERROR - Warp] %s\n", cudaGetErrorString(err));

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    matrixPerformance perf;
    perf.seconds = milliseconds / 1000.0;
    perf.flops = 2.0 * NZ;
    perf.gigaFlops = perf.flops / perf.seconds / 1e9;

    // Cleanup
    cudaFree(d_IRP); cudaFree(d_JA); cudaFree(d_AS);
    cudaFree(d_x);   cudaFree(d_y);
    free(IRP);       free(JA);       free(AS);
    cudaEventDestroy(start); cudaEventDestroy(stop);

    return perf;
}