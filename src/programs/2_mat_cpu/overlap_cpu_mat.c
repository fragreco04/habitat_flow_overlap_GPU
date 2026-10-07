#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "my_utilities.h"

int main(int argc, char *argv[]) {

    CpuTimer total_program_timer = cpu_timer_start();
    CpuTimer pre_processing_timer = cpu_timer_start();

    // Il quarto parametro è una flag che indica se è un warm-up 1 (non salva su json) oppure no 0
    // Il quinto parametro è la destinazione del testing
    if (argc != 5) {
        fprintf(stderr, "Usage: %s <habitat_filepath> <flow_filepath>\n", argv[0]);
        exit(1);
    }

    const char *habi_dir = argv[1];
    if (!check_dir(habi_dir)) {
        fprintf(stderr, "Files in %s must be .txt\n", habi_dir);
        exit(2);
    }

    const char *flow_dir = argv[2];
    if (!check_dir(flow_dir)) {
        fprintf(stderr, "Files in %s must be .txt\n", flow_dir);
        exit(3);
    }

    FileList habi_list = get_filepaths_list(habi_dir);
    FileList flow_list = get_filepaths_list(flow_dir);

    if (habi_list.paths == NULL) {
        fprintf(stderr, "Error: read habitat files\n");
        free_file_list(&habi_list);
        free_file_list(&flow_list);
        exit(7);
    }
    if (flow_list.paths == NULL) {
        fprintf(stderr, "Error: read flow files\n");
        free_file_list(&habi_list);
        free_file_list(&flow_list);
        exit(8);
    }
    if (habi_list.count == 0) {
        fprintf(stderr, "Habitats directory is empty\n");
        free_file_list(&habi_list);
        free_file_list(&flow_list);
        exit(5);
    }
    if (flow_list.count == 0) {
        fprintf(stderr, "Flows directory is empty\n");
        free_file_list(&habi_list);
        free_file_list(&flow_list);
        exit(6);
    }

    printf("\nHabitat files:\n");
    for (size_t i = 0; i < habi_list.count; ++i) {
        printf("\t%s\n", habi_list.paths[i]);
    }
    printf("\nFlow files:\n");
    for (size_t i = 0; i < flow_list.count; ++i) {
        printf("\t%s\n", flow_list.paths[i]);
    }
    printf("\nHabitat: %zu - Flow: %zu\n", habi_list.count, flow_list.count);

    Raster **habi_raster = calloc(habi_list.count, sizeof(Raster *));
    Raster **flow_raster = calloc(flow_list.count, sizeof(Raster *));

    if (habi_raster == NULL || flow_raster == NULL) {
        fprintf(stderr, "Errore allocazione memoria raster array\n");
        free(habi_raster);
        free(flow_raster);
        free_file_list(&habi_list);
        free_file_list(&flow_list);
        exit(11);
    }

    for (size_t i = 0; i < habi_list.count; ++i) {
        habi_raster[i] = load_data_from_txt(habi_list.paths[i]);
        if (habi_raster[i] == NULL) {
            fprintf(stderr, "Errore nel caricamento di %s\n", habi_list.paths[i]);
            free_raster_list(habi_raster, habi_list.count);
            free_raster_list(flow_raster, flow_list.count);
            free_file_list(&habi_list);
            free_file_list(&flow_list);
            exit(9);
        }
    }

    for (size_t i = 0; i < flow_list.count; ++i) {
        flow_raster[i] = load_data_from_txt(flow_list.paths[i]);
        if (flow_raster[i] == NULL) {
            fprintf(stderr, "Errore nel caricamento di %s\n", flow_list.paths[i]);
            free_raster_list(habi_raster, habi_list.count);
            free_raster_list(flow_raster, flow_list.count);
            free_file_list(&habi_list);
            free_file_list(&flow_list);
            exit(10);
        }
    }

    if (!check_same_dimensions(habi_raster, habi_list.count, flow_raster, flow_list.count)) {
        fprintf(stderr, "Matrix must have same sizes\n");
        free_raster_list(habi_raster, habi_list.count);
        free_raster_list(flow_raster, flow_list.count);
        free_file_list(&habi_list);
        free_file_list(&flow_list);
        exit(10);
    }

    double ms_preprocessing = cpu_timer_stop_ms(pre_processing_timer);

    const size_t n = habi_list.count;
    const size_t m = flow_list.count;
    const size_t total_cells = (size_t)habi_raster[0]->rows * (size_t)habi_raster[0]->cols;

    size_t *active_h = calloc(n, sizeof(size_t));
    float *overlap_matrix = calloc(m * n, sizeof(float));

    if (active_h == NULL || overlap_matrix == NULL) {
        fprintf(stderr, "Errore allocazione strutture dati di calcolo\n");
        free(active_h);
        free(overlap_matrix);
        free_raster_list(habi_raster, habi_list.count);
        free_raster_list(flow_raster, flow_list.count);
        free_file_list(&habi_list);
        free_file_list(&flow_list);
        exit(12);
    }

    CpuTimer overlap_timer = cpu_timer_start();

    // Calcolo celle attive di H
    for (size_t j = 0; j < n; ++j) {
        const float *data_h = habi_raster[j]->data;
        size_t count = 0;
        for (size_t k = 0; k < total_cells; ++k) {
            if (data_h[k] != 0.0f) {
                count++;
            }
        }
        active_h[j] = count;
    }

    // Calcolo celle attive di H e F
    for (size_t i = 0; i < m; ++i) {
        const float *data_f = flow_raster[i]->data;

        for (size_t j = 0; j < n; ++j) {
            if (active_h[j] == 0) {
                overlap_matrix[i * n + j] = 0.0f;
                continue;
            }

            const float *data_h = habi_raster[j]->data;
            size_t intersection_count = 0;

            for (size_t k = 0; k < total_cells; ++k) {
                if (data_f[k] != 0.0f && data_h[k] != 0.0f) {
                    intersection_count++;
                }
            }

            overlap_matrix[i * n + j] = (float)intersection_count / (float)active_h[j];
        }
    }

    double ms_overlap = cpu_timer_stop_ms(overlap_timer);


    printf("\n");
    printf("Matrice di Overlap (Righe: Flow %zu, Colonne: Habitat %zu)\n", m, n);
    printf("\n");

    for (size_t j = 0; j < n; ++j) {
        for (size_t i = 0; i < m; ++i) {
            printf("%.4f\t", overlap_matrix[i * n + j]);
        }
        printf("\n");
    }

    // -----------------------------------------------------------

    char *json_filepath = argv[4];
    if (strcmp(json_filepath, "no") == 0) {
        printf("No salvataggio come test.\n");
    }
    else {
        int is_warm_up = atoi(argv[3]);

        double json_kernel = ms_overlap;
        double json_useful_threads = (double)total_cells * (double)(n * m);
    
        int version = 1;
    
        if (is_warm_up != 1) {
            if (append_benchmark_to_json(json_filepath, version, json_kernel, json_useful_threads) == 0) {
                printf("\nBenchmark salvato con successo in %s [versione %d, vec %zu]\n", json_filepath, version, (size_t)1);
            } else {
                fprintf(stderr, "\nErrore durante il salvataggio in %s\n", json_filepath);
            }
        }
    }

    // -----------------------------------------------------------

    free(active_h);
    free(overlap_matrix);
    free_raster_list(habi_raster, habi_list.count);
    free_raster_list(flow_raster, flow_list.count);
    free_file_list(&habi_list);
    free_file_list(&flow_list);

    double ms_total_program = cpu_timer_stop_ms(total_program_timer);

    printf("\n");
    printf("--- TEMPI DEL PROGRAMMA ---\n");
    printf("Tempo pre-processing:   %8.2f ms\n", ms_preprocessing);
    printf("Tempo overlap:          %8.2f ms\n", ms_overlap);
    printf("Tempo totale:           %8.2f ms\n", ms_total_program);


    return 0;
}