#include <ctype.h>
#include <omp.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <sys/types.h>
#include "/home/luigi/SCPA_Progetto/libs/mmio.h"
#include "/home/luigi/SCPA_Progetto/libs/data_structure.h"
#include "/home/luigi/SCPA_Progetto/libs/matrixLists.h"
#include "/home/luigi/SCPA_Progetto/libs/csr_utils.h"

const char *base_path = "/home/luigi/SCPA_Progetto/matrix/";
// Prodotto matrice-vettore CSR seriale
void csr_matrix_vector_product(int M, const int *IRP, const int *JA, const double *AS, const double *x, double *y) {
    for (int i = 0; i < M; i++) {
        double sum = 0.0;
        for (int j = IRP[i]; j < IRP[i + 1]; j++) {
            sum += AS[j] * x[JA[j]];
        }
        y[i] = sum;
    }
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
    const int num_matrices = sizeof(matrix_names) / sizeof(matrix_names[0]);

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

        // Alloca CSR
        int *IRP, *JA;
        double *AS;
        convert_to_csr(matrix_data->M, matrix_data->nz, matrix_data->row_indices, matrix_data->col_indices, matrix_data->values, &IRP, &JA, &AS);

        // Alloca vettore risultato
        double *y = malloc(matrix_data->M * sizeof(double));
        if (!y) {
            perror("Errore allocazione vettore y");
            return EXIT_FAILURE;
        }

        // Esegui prodotto CSR 50 volte
        for (int j = 0; j < 50; j++) {
            csr_matrix_vector_product(matrix_data->M, IRP, JA, AS, x, y);
        }

        printf("✅ Preprocessamento e prodotto CSR completati.\n");
        printf("   Dimensione: %d x %d, NZ: %d\n", matrix_data->M, matrix_data->N, matrix_data->nz);

        // Libera memoria
        free(matrix_data->row_indices);
        free(matrix_data->col_indices);
        free(matrix_data->values);
        free(matrix_data);
        free(x);
        free(y);
        free(IRP);
        free(JA);
        free(AS);
    }

    return 0;
}
    