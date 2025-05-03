#include <ctype.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "../libs/mmio.h"
#include "../libs/data_structure.h"
#include "../libs/matrixLists.h"
#include "../libs/csr_utils.h"
#include "../libs/csr_Operations.h"
#include "../libs/hll_ellpack_utils.h" 

const char *base_path = "matrix/";  

struct matrixPerformance benchmark(
    struct matrixData *matrix_data,
    double *x,
    int num_iter,
    int num_threads,
    MatVecKernel kernel_func
) {
    struct matrixPerformance result = {0};

    for (int i = 0; i < num_iter; i++) {
        struct matrixPerformance temp = kernel_func(matrix_data, x, num_threads);
        result.seconds += temp.seconds;
        result.flops += temp.flops;
        result.gigaFlops += temp.gigaFlops;
    }

    result.seconds /= num_iter;
    result.flops /= num_iter;
    result.gigaFlops /= num_iter;

    return result;
}

/* Funzione per il preprocessamento delle matrici in input da file */
void preprocess_matrix(struct matrixData *matrix_data, int i) {
    char full_path[512];
    snprintf(full_path, sizeof(full_path), "%s%s", base_path, matrix_names[i]);

    FILE *f = fopen(full_path, "r");
    if (!f) {
        perror("Errore nell'apertura del file");
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

    matrix_data->row_indices = malloc(nz * sizeof(int));
    matrix_data->col_indices = malloc(nz * sizeof(int));
    matrix_data->values = malloc(nz * sizeof(double));
    matrix_data->M = M;
    matrix_data->N = N;
    matrix_data->nz = nz;

    if (matrix_data->row_indices == NULL || matrix_data->col_indices == NULL || matrix_data->values == NULL || matrix_data->M == 0 || matrix_data->N == 0 || matrix_data->nz == 0) {
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
        matrix_data->row_indices = realloc(matrix_data->row_indices, (nz + extra_nz) * sizeof(int));
        matrix_data->col_indices = realloc(matrix_data->col_indices, (nz + extra_nz) * sizeof(int));
        matrix_data->values = realloc(matrix_data->values, (nz + extra_nz) * sizeof(double));

        if (matrix_data->row_indices == NULL || matrix_data->col_indices == NULL || matrix_data->values == NULL) {
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

    int max_threads = omp_get_max_threads();
    int num_configs = max_threads; // da 1 a max_threads
    const int num_matrices = sizeof(matrix_names) / sizeof(matrix_names[0]);
     // Apri due file separati
     FILE *csr_serial = fopen("results_local/csr_serial.csv", "w");
     FILE *csr_parallel = fopen("results_local/csr_parallel.csv", "w");
     FILE *hll_serial = fopen("results_local/hll_serial.csv", "w");  
     FILE *hll_parallel = fopen("results_local/hll_parallel.csv", "w");
     if (!csr_serial || !csr_parallel|| !hll_serial|| !hll_parallel) {
         perror("Errore apertura file CSV");
         return EXIT_FAILURE;
     }

    // Intestazioni
    fprintf(csr_serial, "Matrice,M,N,NZ,Densità,Thread,Tempo Medio (s),FLOPs,GFLOPS\n");
    fprintf(csr_parallel, "Matrice,M,N,NZ,Densità,Thread,Tempo Medio (s),FLOPs,GFLOPS,Speedup\n");
    fprintf(hll_serial, "Matrice,M,N,NZ,Densità,Thread,Tempo Medio (s),FLOPs,GFLOPS\n");
    fprintf(hll_parallel, "Matrice,M,N,NZ,Densità,Thread,Tempo Medio (s),FLOPs,GFLOPS,Speedup\n");

    for (int i = 0; i < num_matrices; i++) {
        printf("\n--- Matrice: %s ---\n", matrix_names[i]);

        struct matrixData *matrix_data = malloc(sizeof(struct matrixData));
        if (!matrix_data) {
            perror("Errore nell'allocazione di matrix_data");
            return EXIT_FAILURE;
        }

        preprocess_matrix(matrix_data, i);

        // Alloca e inizializza vettore x
        double *x = malloc(matrix_data->N * sizeof(double));
        if (!x) {
            perror("Errore allocazione vettore x");
            free(matrix_data);
            return EXIT_FAILURE;
        }
        for (int j = 0; j < matrix_data->N; j++) {
            x[j] = 1.0;
        }

        // Alloca vettore risultato
        double *y = malloc(matrix_data->M * sizeof(double));
        if (!y) {
            perror("Errore allocazione vettore y");
            return EXIT_FAILURE;
        }
        int nz = matrix_data->nz;
        double density = (double)nz / (matrix_data->M * matrix_data->N);
        double flops = 2.0 * nz;

        // Esegui prodotto  50 volte cambiato in 3 
        int ITERATION = 1;
        struct matrixPerformance perf_serial_csr = benchmark(matrix_data, x, ITERATION, 1, serial_csr);
        fprintf(csr_serial, "%s,%d,%d,%d,%.8f,%d,%.6f,%.0f,%.6f\n",
            matrix_names[i],
            matrix_data->M,
            matrix_data->N,
            nz,
            density,
            1, // thread
            perf_serial_csr.seconds,
            flops,
            perf_serial_csr.gigaFlops);
  
        
        for (int threads = 2; threads <= max_threads; threads++) {
            struct matrixPerformance perf_parallel_csr = benchmark(matrix_data, x, ITERATION, threads, parallel_csr);
             // Calcola speedup (solo se flops seriali > 0)
            double speedup = (perf_serial_csr.gigaFlops > 0) ?
            (perf_parallel_csr.gigaFlops / perf_serial_csr.gigaFlops) : 0.0;
            fprintf(csr_parallel, "%s,%d,%d,%d,%.8f,%d,%.6f,%.0f,%.6f,%.2f\n",
                matrix_names[i],
                matrix_data->M,
                matrix_data->N,
                nz,
                density,
                threads,
                perf_parallel_csr.seconds,
                flops,
                perf_parallel_csr.gigaFlops,
                speedup);
            }

        struct matrixPerformance perf_serial_hll = benchmark(matrix_data, x, ITERATION, 1, serial_hll);
        
   
        fprintf(hll_serial, "%s,%d,%d,%d,%.8f,%d,%.6f,%.0f,%.6f\n",
                matrix_names[i],
                matrix_data->M,
                matrix_data->N,
                nz,
                density,
                1,  // thread
                perf_serial_hll.seconds,
                flops,
                perf_serial_hll.gigaFlops);

            for (int threads = 2; threads <= max_threads; threads++) {
                struct matrixPerformance perf_parallel_hll = benchmark(matrix_data, x, ITERATION, threads, parallel_hll);
                double speedup = (perf_serial_hll.gigaFlops > 0) ?
                                    (perf_parallel_hll.gigaFlops / perf_serial_hll.gigaFlops) : 0.0;
            
                fprintf(hll_parallel, "%s,%d,%d,%d,%.8f,%d,%.6f,%.0f,%.6f,%.2f\n",
                        matrix_names[i],
                        matrix_data->M,
                        matrix_data->N,
                        nz,
                        density,
                        threads,
                        perf_parallel_hll.seconds,
                        flops,
                        perf_parallel_hll.gigaFlops,
                        speedup);
            }
                
       
 
        // Libera memoria
        free(matrix_data->row_indices);
        free(matrix_data->col_indices);
        free(matrix_data->values);
        free(matrix_data);
        free(x);
        free(y);
    
    }
    fclose(csr_serial);
    fclose(csr_parallel);
    fclose(hll_serial);  // chiudi anche questo
    fclose(hll_parallel); 
    return 0;
}
    