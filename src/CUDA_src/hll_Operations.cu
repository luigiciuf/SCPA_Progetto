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
    double *d_y = nullptr, *d_x = nullptr;
    int M = matrix_data_host->M;
    int maxThreadsPerBlock = 0, maxGridDimX = 0, grid_x = 0, grid_y = 0, numBlock = 0;
    int max_nz_per_row_global = 0;

    // Resetta e opzionalmente controlla quanta memoria è disponibile
    cudaDeviceReset();
    //cudaMemGetInfo(nullptr, nullptr); // opzionale

    // Alloca la struttura HLL su host
    HLL_Matrix *hllMatrixHost = (HLL_Matrix *)malloc(sizeof(HLL_Matrix));
    if (!hllMatrixHost) {
        fprintf(stderr, "Errore allocazione HLL_Matrix\n");
        exit(EXIT_FAILURE);
    }

    hllMatrixHost->num_blocks = (M + HackSize - 1) / HackSize;
    hllMatrixHost->blocks = (ELLPACK_Block *)malloc(hllMatrixHost->num_blocks * sizeof(ELLPACK_Block));
    if (!hllMatrixHost->blocks) {
        fprintf(stderr, "Errore allocazione blocchi ELLPACK\n");
        free(hllMatrixHost);
        exit(EXIT_FAILURE);
    }

    // Vettore y su host
    double *y_h = (double *)malloc(M * sizeof(double));
    if (!y_h) {
        fprintf(stderr, "Errore allocazione vettore y\n");
        // Libera quello che hai già allocato
        free(hllMatrixHost->blocks);
        free(hllMatrixHost);
        exit(EXIT_FAILURE);
    }
    memset(y_h, 0, M * sizeof(double)); // inizialmente a 0

    // Converte la matrice in formato HLL su host (alloca JA e AS con calloc)
    convert_to_hll_cuda(matrix_data_host, hllMatrixHost);

    // Alloca la struttura HLL su device (puntatore "d_hll_matrix")
    HLL_Matrix *d_hll_matrix = nullptr;
    cudaMalloc(&d_hll_matrix, sizeof(HLL_Matrix));
    cudaMemcpy(d_hll_matrix, hllMatrixHost, sizeof(HLL_Matrix), cudaMemcpyHostToDevice);

    // Alloca array di blocchi su device
    ELLPACK_Block *d_blocks = nullptr;
    cudaMalloc(&d_blocks, hllMatrixHost->num_blocks * sizeof(ELLPACK_Block));

    // Copia il puntatore d_blocks nella struttura HLL su device
    // (cioè, in d_hll_matrix->blocks va scritto l'indirizzo "d_blocks")
    cudaMemcpy(&(d_hll_matrix->blocks), &d_blocks, sizeof(ELLPACK_Block*), cudaMemcpyHostToDevice);

    // Copia i dati di ciascun blocco su device
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        ELLPACK_Block *block = &hllMatrixHost->blocks[i];

        // Calcola la dimensione in byte dei vettori del blocco
        size_t JA_size = block->size_of_arrays * sizeof(int);
        size_t AS_size = block->size_of_arrays * sizeof(double);

        // Alloca e copia JA su device
        int *d_JA = nullptr;
        cudaMalloc(&d_JA, JA_size);
        cudaMemcpy(d_JA, block->JA, JA_size, cudaMemcpyHostToDevice);

        // Alloca e copia AS su device
        double *d_AS = nullptr;
        cudaMalloc(&d_AS, AS_size);
        cudaMemcpy(d_AS, block->AS, AS_size, cudaMemcpyHostToDevice);

        // Crea un blocco "temporaneo" su host, per passarlo al device
        ELLPACK_Block d_block;
        d_block.JA             = d_JA;
        d_block.AS             = d_AS;
        d_block.max_nz_per_row = block->max_nz_per_row;
        d_block.nz_per_block   = block->nz_per_block;
        d_block.size_of_arrays = block->size_of_arrays;

        // Copia il d_block in d_blocks[i]
        cudaMemcpy(&d_blocks[i], &d_block, sizeof(ELLPACK_Block), cudaMemcpyHostToDevice);
    }

    // Alloca x e y su device
    cudaMalloc(&d_x, M * sizeof(double));
    cudaMemcpy(d_x, x_h, M * sizeof(double), cudaMemcpyHostToDevice);

    cudaMalloc(&d_y, M * sizeof(double));
    cudaMemset(d_y, 0, M * sizeof(double));

    // Recupera caratteristiche hardware
    cudaDeviceGetAttribute(&maxThreadsPerBlock, cudaDevAttrMaxThreadsPerBlock, 0);
    cudaDeviceGetAttribute(&maxGridDimX, cudaDevAttrMaxGridDimX, 0);

    // Determina numBlock e max_nz_per_row_global
    numBlock = hllMatrixHost->num_blocks;
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        if (hllMatrixHost->blocks[i].max_nz_per_row > max_nz_per_row_global)
            max_nz_per_row_global = hllMatrixHost->blocks[i].max_nz_per_row;
    }

    // Allinea max_nz_per_row_global e numBlock a multipli di HackSize
    if (max_nz_per_row_global % HackSize != 0)
        max_nz_per_row_global = ((max_nz_per_row_global / HackSize) + 1) * HackSize;

    if (max_nz_per_row_global > maxThreadsPerBlock / HackSize)
        max_nz_per_row_global = maxThreadsPerBlock / HackSize;

    if (numBlock % HackSize != 0)
        numBlock = ((numBlock / HackSize) + 1) * HackSize;

    if (numBlock > maxGridDimX)
        numBlock = maxGridDimX;

    // Imposta la griglia
    grid_x = (int)sqrt((float)numBlock);
    grid_y = (numBlock + grid_x - 1) / grid_x;

    dim3 BLOCK_DIM(HackSize, HackSize);
    dim3 GRID_DIM(grid_x, grid_y);

    // === Timing con cudaEvent ===
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Lancio del kernel
    matvec_Hll_cuda_SH<<<GRID_DIM, BLOCK_DIM>>>(d_hll_matrix, d_x, d_y, M);
    cudaDeviceSynchronize();

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // === Copia risultati y indietro su host
    cudaMemcpy(y_h, d_y, M * sizeof(double), cudaMemcpyDeviceToHost);

    // --- Ora rilasciamo le risorse GPU ---

    // 1) Libera i vettori di ciascun blocco su device
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        ELLPACK_Block tmp_block;
        // Copio il blocco device in una struttura host temporanea
        cudaMemcpy(&tmp_block, &d_blocks[i], sizeof(ELLPACK_Block), cudaMemcpyDeviceToHost);

        // Libero i puntatori device
        cudaFree(tmp_block.JA);
        cudaFree(tmp_block.AS);
    }

    // 2) Libera l'array di blocchi e la struttura HLL su device
    cudaFree(d_blocks);
    cudaFree(d_hll_matrix);

    // 3) Libera x e y su device
    cudaFree(d_x);
    cudaFree(d_y);

    // --- Adesso rilasciamo la memoria host ---
    // I blocchi su host sono stati allocati con calloc(...) in convert_to_hll_cuda
    for (int i = 0; i < hllMatrixHost->num_blocks; i++) {
        free(hllMatrixHost->blocks[i].JA);
        free(hllMatrixHost->blocks[i].AS);
    }
    free(hllMatrixHost->blocks);
    free(hllMatrixHost);

    // Libera il vettore y su host
    free(y_h);

    // Distrugge gli event di timing
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Calcolo performance
    matrixPerformance perf{};
    perf.seconds = milliseconds / 1000.0;
    perf.flops = 2.0 * matrix_data_host->nz;
    perf.gigaFlops = perf.flops / perf.seconds / 1e9;

    return perf;
}
