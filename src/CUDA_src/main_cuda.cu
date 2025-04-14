#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include  "../CUDA_libs/csr_Operation.h"
#include  "../CUDA_libs/hll_Operations.h"

#include "../../libs/data_structure.h"
#include "../../libs/matrixLists.h"
#include "../../libs/mmio.h"
#include "../../libs/costants.h"
#include "../../src/matrixIO.c"

const char *base_path = "matrix/";
/* Funzione per il preprocessamento delle matrici in input da file */
void preprocess_matrix(matrixData *matrix_data, int i) {
    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s%s", base_path, matrix_names[i]);

    FILE *f = fopen(full_path, "r");
    if (!f) {
        perror("Errore nell'apertura del file della matrice");
        exit(EXIT_FAILURE);
    }

    /* Lettura dell'intestazione (banner) del file Matrix Market */
    if (mm_read_banner(f, &matrix_data->matcode) != 0) {
        fprintf(stderr, "Errore nella lettura del banner Matrix Market.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    /* Verifica del formato della matrice */
    if (!mm_is_matrix(matrix_data->matcode) || !mm_is_coordinate(matrix_data->matcode)) {
        fprintf(stderr, "Il file non è in formato matrice sparsa a coordinate.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    /* Lettura dei parametri dimensionali della matrice */
    int M, N, nz;
    if (mm_read_mtx_crd_size(f, &M, &N, &nz) != 0) {
        fprintf(stderr, "Errore nella lettura delle dimensioni della matrice.\n");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    matrix_data->row_indices = static_cast<int *>(malloc(nz * sizeof(int)));
    matrix_data->col_indices = static_cast<int *>(malloc(nz * sizeof(int)));
    matrix_data->values = static_cast<double *>(malloc(nz * sizeof(double)));
    matrix_data->M = M;
    matrix_data->N = N;
    matrix_data->nz = nz;

    if (matrix_data->row_indices == nullptr || matrix_data->col_indices == nullptr || matrix_data->values == nullptr || matrix_data->M == 0 || matrix_data->N == 0 || matrix_data->nz == 0) {
        perror("Errore nell'allocazione della memoria.");
        fclose(f);
        exit(EXIT_FAILURE);
    }

    /* contiene preprocessamento matrice pattern */
    for (int j = 0; j < nz; j++) {
        int result;
        double value = 1.0; // Valore predefinito per matrici "pattern"

        if (mm_is_pattern(matrix_data->matcode)) {   // Preprocessamento matrice in formato pattern
            result = fscanf(f, "%d %d", &matrix_data->row_indices[j], &matrix_data->col_indices[j]);
        } else {
            result = fscanf(f, "%d %d %lf", &matrix_data->row_indices[j], &matrix_data->col_indices[j], &value);
        }

        if (result != (mm_is_pattern(matrix_data->matcode) ? 2 : 3)) {
            fprintf(stderr, "Errore nella lettura degli elementi della matrice.\n");
            free(matrix_data->row_indices);
            free(matrix_data->col_indices);
            free(matrix_data->values);
            fclose(f);
            exit(EXIT_FAILURE);
        }

        matrix_data->row_indices[j]--; // Converti a indice 0-based
        matrix_data->col_indices[j]--; // Converti a indice 0-based
        matrix_data->values[j] = value;
    }

    /* Preprocessamento matrice simmetrica */
    if (mm_is_symmetric(matrix_data->matcode)) {
        int extra_nz = 0;
        for (int j = 0; j < nz; j++) {
            if (matrix_data->row_indices[j] != matrix_data->col_indices[j]) {
                extra_nz++;
            }
        }

        // Estensione degli array con il numero di non zeri da aggiungere
        matrix_data->row_indices = static_cast<int *>(realloc(matrix_data->row_indices, (nz + extra_nz) * sizeof(int)));
        matrix_data->col_indices = static_cast<int *>(realloc(matrix_data->col_indices, (nz + extra_nz) * sizeof(int)));
        matrix_data->values = static_cast<double *>(realloc(matrix_data->values, (nz + extra_nz) * sizeof(double)));

        if (matrix_data->row_indices == nullptr || matrix_data->col_indices == nullptr || matrix_data->values == nullptr) {
            perror("Errore nell'allocazione della memoria.");
            fclose(f);
            exit(EXIT_FAILURE);
        }

        // Aggiunta degli elementi simmetrici
        int index = nz;
        for (int j = 0; j < nz; j++) {
            if (matrix_data->row_indices[j] != matrix_data->col_indices[j]) {
                matrix_data->row_indices[index] = matrix_data->col_indices[j];
                matrix_data->col_indices[index] = matrix_data->row_indices[j];
                matrix_data->values[index] = matrix_data->values[j];
                index++;
            }
        }
        matrix_data->nz += extra_nz;
    }
    fclose(f);
}

int main() {
    const int num_matrices = sizeof(matrix_names) / sizeof(matrix_names[0]);
    const int ITERATION = 50;

    FILE *cuda_perf_csv = fopen("results_local/cuda_performance.csv", "w");
    if (!cuda_perf_csv) {
        perror("Errore nell'apertura del file cuda_performance.csv");
        return EXIT_FAILURE;
    }

    // Intestazione CSV unica
    fprintf(cuda_perf_csv, "Matrice,M,N,NZ,Densità,GFLOPS_CSR_Seriale,GFLOPS_CSR_Parallelo,GFLOPS_CSR_WARP,GFLOPS_HLL_Seriale\n");

    for (int i = 0; i < num_matrices; i++) {
        printf("\n--- Matrice: %s ---\n", matrix_names[i]);

        matrixData *matrix_data = static_cast<matrixData *>(malloc(sizeof(matrixData)));
        if (!matrix_data) {
            perror("Errore nell'allocazione di matrix_data");
            return EXIT_FAILURE;
        }

        preprocess_matrix(matrix_data, i);

        double *x = static_cast<double *>(malloc(matrix_data->N * sizeof(double)));
        if (!x) {
            perror("Errore allocazione vettore x");
            return EXIT_FAILURE;
        }
        for (int j = 0; j < matrix_data->N; j++) {
            x[j] = 1.0;
        }

        int nz = matrix_data->nz;
        double density = (double)nz / (matrix_data->M * matrix_data->N);

        // ==== CSR Serial CUDA ====
        double total_gflops_csr = 0.0;
        for (int k = 0; k < ITERATION; k++) {
            matrixPerformance perf = serial_csr_cuda(matrix_data, x);
            total_gflops_csr += perf.gigaFlops;
        }
        double avg_gflops_csr = total_gflops_csr / ITERATION;

        // ==== CSR Parallel CUDA ====
        double total_gflops_parallel = 0.0;
        for (int k = 0; k < ITERATION; k++) {
            matrixPerformance perf = parallel_csr_cuda(matrix_data, x);
            total_gflops_parallel += perf.gigaFlops;
        }
        double avg_gflops_parallel = total_gflops_parallel / ITERATION;

        // ==== CSR Warp CUDA ====
        double total_gflops_warp = 0.0;
        for (int k = 0; k < ITERATION; k++) {
            matrixPerformance perf = parallel_csr_cuda_warp(matrix_data, x);
            total_gflops_warp += perf.gigaFlops;
        }
        double avg_gflops_warp = total_gflops_warp / ITERATION;

        // ==== HLL Serial CUDA ====
        double total_gflops_hll = 0.0;
        for (int k = 0; k < ITERATION; k++) {
            matrixPerformance perf = serial_hll_cuda(matrix_data, x);
            total_gflops_hll += perf.gigaFlops;
        }
        double avg_gflops_hll = total_gflops_hll / ITERATION;

        // Scrivi una riga nel file unificato
        fprintf(cuda_perf_csv, "%s,%d,%d,%d,%.8f,%.6f,%.6f,%.6f,%.6f\n",
                matrix_names[i], matrix_data->M, matrix_data->N, nz,
                density, avg_gflops_csr, avg_gflops_parallel, avg_gflops_warp, avg_gflops_hll);

        // Cleanup
        free(matrix_data->row_indices);
        free(matrix_data->col_indices);
        free(matrix_data->values);
        free(matrix_data);
        free(x);
    }

    fclose(cuda_perf_csv);

    return 0;
}