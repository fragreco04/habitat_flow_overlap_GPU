#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "ocl_boiler.h"
#include "my_utilities.h"
#include "../info.h"

cl_event overlap(
    cl_command_queue que,
    cl_kernel kernel,
    cl_mem all_habitats,
    cl_mem all_flows,
    cl_int num_vectors,
    cl_int m_flows,
    cl_mem output,
    size_t *lws,
    size_t *gws,
    cl_uint num_evt,
    cl_event *event_wait_list
) {
    cl_int err;
    cl_event overlap_evt;

    size_t local_mem_bytes = lws[0] * sizeof(cl_uint);
    cl_uint arg_index = 0;

    err = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &all_habitats);
    ocl_check(err, "init_k set kernel arg %d", arg_index);
    ++arg_index;

    err = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &all_flows);
    ocl_check(err, "init_k set kernel arg %d", arg_index);
    ++arg_index;

    err = clSetKernelArg(kernel, arg_index, sizeof(cl_int), &num_vectors);
    ocl_check(err, "init_k set kernel arg %d", arg_index);
    ++arg_index;

    err = clSetKernelArg(kernel, arg_index, sizeof(cl_int), &m_flows);
    ocl_check(err, "init_k set kernel arg %d", arg_index);
    ++arg_index;

    err = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &output);
    ocl_check(err, "init_k set kernel arg %d", arg_index);
    ++arg_index;

    err = clSetKernelArg(kernel, arg_index, local_mem_bytes, NULL);
    ocl_check(err, "init_k set kernel arg %d", arg_index);
    ++arg_index;

    err = clSetKernelArg(kernel, arg_index, local_mem_bytes, NULL);
    ocl_check(err, "init_k set kernel arg %d", arg_index);
    ++arg_index;

    err = clEnqueueNDRangeKernel(
        que, kernel,
        2, NULL,
        gws, lws,
        num_evt, event_wait_list,
        &overlap_evt
    );

    ocl_check(err, "enqueue kernel overlap_k");
    return overlap_evt;
}

int main(int argc, char *argv[]) {
    
    CpuTimer total_program_timer = cpu_timer_start();
    CpuTimer pre_processing_timer = cpu_timer_start();

    // Il quinto parametro è una flag che indica se è un warm-up 1 (non salva su json) oppure no 0
    // Il sesto parametro è la destinazione del testing
    if (argc != 6) {
        fprintf(stderr, "Usage: %s <habitat_filepath> <flow_filepath> <vec>\n", argv[0]);
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

    const size_t vec = atoi(argv[3]);
    if (check_vec(vec) == 0) {
        fprintf(stderr, "Invalid value of vec (1, 2, 4, 8, 16)\n");
        exit(4);
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

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    printf("\n");

    double ms_preprocessing = cpu_timer_stop_ms(pre_processing_timer);

    cl_int err;

    cl_program prog = create_program("overlap.ocl", ctx, d);

    // Impostiamo il kernel
    char *kernel_name = get_kernel_name(vec);
    cl_kernel overlap_k = clCreateKernel(prog, kernel_name, &err);
    ocl_check(err, "clCreateKernel %s fallito", kernel_name);

    const size_t nels = habi_raster[0]->total_cells;
    size_t nels_pad = nels;
    if (nels % vec != 0) {
        nels_pad = nels + (vec - (nels % vec));
    }

    cl_uint num_vectors = (cl_uint)(nels_pad / vec);

    size_t memsize_host = sizeof(u_char) * nels;
    size_t memsize_dev  = sizeof(u_char) * nels_pad;

    size_t n = habi_list.count;
    size_t m = flow_list.count;

    GPU_Conf conf;
    gpu_auto_conf(d, overlap_k, nels, vec, n*m, &conf);

    // Prepariamo il trasferimento dei dati sull'host
    u_char *h_all_habitats = calloc(n * nels_pad, sizeof(u_char));
    u_char *h_all_flows = calloc(m * nels_pad, sizeof(u_char));

    if (h_all_habitats == NULL || h_all_flows == NULL) {
        fprintf(stderr, "Error: memory allocation h_all_habitats / h_all_flows");
        exit(12);
    }

    // Prepariamo i dati per il trasferimento in batch
    for (size_t i = 0; i < n; ++i) {
        memcpy(h_all_habitats + (i * nels_pad), habi_raster[i]->data, memsize_host);
    }
    for (size_t j = 0; j < m; ++j) {
        memcpy(h_all_flows + (j * nels_pad), flow_raster[j]->data, memsize_host);
    }

    // Creazione dei buffer GPU
    cl_mem d_all_habitats = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n * memsize_dev, NULL, &err);
    ocl_check(err, "clCreateBuffer failed (d_all_habitats)");
    cl_mem d_all_flows = clCreateBuffer(ctx, CL_MEM_READ_ONLY, m * memsize_dev, NULL, &err);
    ocl_check(err, "clCreateBuffer failed (d_all_flows)");
    
    // Buffer GPU di output, otterremo [h_count, overlap_count] per ogni cella
    cl_mem d_counts = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n * m * 2 * sizeof(cl_uint), NULL, &err);
    ocl_check(err, "clCreateBuffer failed (d_counts)");

    cl_uint zero = 0;
    err = clEnqueueFillBuffer(que, d_counts, &zero, sizeof(cl_uint), 0, n * m * 2 * sizeof(cl_uint), 0, NULL, NULL);
    ocl_check(err, "fill d_counts");

    // Matrice risultante
    float *overlap_matrix = malloc(sizeof(float) * n * m);
    if (overlap_matrix == NULL) { fprintf(stderr, "Error: overlap_matrix allocation"); exit(12); }

    cl_uint *h_counts = malloc(n * m * 2 * sizeof(cl_uint));
    if (h_counts == NULL) { fprintf(stderr, "Error: h_counts allocation"); exit(12); }

    cl_event hab_write_evt = NULL;
    cl_event flow_write_evt = NULL;
    cl_event kernel_evt = NULL;
    cl_event read_evt = NULL;

    err = clEnqueueWriteBuffer(que, d_all_habitats, CL_FALSE, 0, n * memsize_dev, h_all_habitats, 0, NULL, &hab_write_evt);
    ocl_check(err, "write all habitats");

    err = clEnqueueWriteBuffer(que, d_all_flows, CL_FALSE, 0, m * memsize_dev, h_all_flows, 0, NULL, &flow_write_evt);
    ocl_check(err, "write all flows");

    cl_event batch_waits[] = {hab_write_evt, flow_write_evt};
    kernel_evt = overlap(que, overlap_k, d_all_habitats, d_all_flows, num_vectors, (cl_uint)m, d_counts, conf.lws, conf.gws, 2, batch_waits);

    err = clEnqueueReadBuffer(que, d_counts, CL_TRUE, 0, n * m * 2 * sizeof(cl_uint), h_counts, 1, &kernel_evt, &read_evt);
    ocl_check(err, "read all counts");


    CpuTimer cpu_timer = cpu_timer_start();
    for (size_t pair = 0; pair < n * m; ++pair) {
        cl_uint h_val = h_counts[pair * 2];     // pos pari
        cl_uint overlap_val = h_counts[pair * 2 + 1];       // pos dispari
        overlap_matrix[pair] = (h_val > 0) ? ((float)overlap_val / (float)h_val) : 0.0f;
    }
    double ms_total_cpu = cpu_timer_stop_ms(cpu_timer);
    
    printf("\n");
    print_matrix(overlap_matrix, n, m);

    cl_ulong write_hab_ns = runtime_ns(hab_write_evt);
    cl_ulong write_flow_ns = runtime_ns(flow_write_evt);
    cl_ulong kernel_ns = runtime_ns(kernel_evt);
    cl_ulong read_ns = runtime_ns(read_evt);

    cl_ulong total_write_ns  = write_hab_ns + write_flow_ns;

    // Byte transitati sul bus pcie 
    double bytes_h2d_pcie = (double)memsize_dev * (double)(n + m);
    double bytes_d2h_pcie = (double)(n * m) * (double)(2 * sizeof(cl_uint));

    // Byte elaborati dalla DRAM della GPU (senza contare il padding)
    double total_kernel_read_bytes = 2.0 * (double)memsize_host * (double)(n * m);
    double total_kernel_write_bytes = bytes_d2h_pcie;

    double effective_bw = ((total_kernel_read_bytes + total_kernel_write_bytes) / (kernel_ns * 1.0e-9)) / 1.0e9;
    double bw_efficiency = (peak_memory_bandwidth > 0.0) 
                           ? (effective_bw / peak_memory_bandwidth) * 100.0 
                           : 0.0;

    printf("\n");
    printf("==================== REPORT PRESTAZIONALE ====================\n");
    printf("Configurazione Griglia: GWS [%zu, %zu] | LWS [%zu, %zu]\n", 
           conf.gws[0], conf.gws[1], conf.lws[0], conf.lws[1]);
    printf("Grid Efficiency:       %8.2f %% (Thread utili: %zu / Totali: %zu)\n", 
           (double)(conf.grid_efficiency * 100), conf.useful_threads, conf.gws[0]*conf.gws[1]);
    printf("--------------------------------------------------------------\n");
    printf("Trasferimento H2D:     %8.2f ms | Banda PCIe: %6.2f GB/s\n", 
           total_write_ns * 1.0e-6, bytes_h2d_pcie / (double)total_write_ns);
    printf("Esecuzione Kernel:     %8.2f ms\n", 
           kernel_ns * 1.0e-6);
    printf("Trasferimento D2H:     %8.2f ms | Banda PCIe: %6.2f GB/s\n", 
           read_ns * 1.0e-6, bytes_d2h_pcie / (double)read_ns);
    printf("--------------------------------------------------------------\n");
    printf("Banda DRAM Effettiva:  %8.2f GB/s\n", effective_bw);
    printf("Banda DRAM Teorica:    %8.2f GB/s\n", peak_memory_bandwidth);
    printf("Efficienza di Memoria: %8.2f %%\n", bw_efficiency);
    printf("==============================================================\n\n");

    // -----------------------------------------------------------

    char *json_filepath = argv[5];
    if (strcmp(json_filepath, "no") == 0) {
        printf("No salvataggio come test.\n");
    }
    else {
        int is_warm_up = atoi(argv[4]);

        double json_kernel = (double)kernel_ns * 1.0e-6;
        double json_h2d = (double)total_write_ns * 1.0e-6;
        double json_d2h = (double)read_ns * 1.0e-6;
        double json_cpu_post = ms_total_cpu;
        double json_effective_bw = effective_bw;
        double json_useful_threads = conf.useful_threads;
    
        int version = 4;
    
        if (is_warm_up != 1) {
            if (append_benchmark_to_json(json_filepath, version, vec, json_kernel, json_h2d, json_d2h, json_cpu_post, json_effective_bw, json_useful_threads) == 0) {
                printf("\nBenchmark salvato con successo in %s [versione %d, vec %zu]\n", json_filepath, version, vec);
            } else {
                fprintf(stderr, "\nErrore durante il salvataggio in %s\n", json_filepath);
            }
        }
    }

    // -----------------------------------------------------------

    clFinish(que);

    clReleaseEvent(hab_write_evt);
    clReleaseEvent(flow_write_evt);
    clReleaseEvent(kernel_evt);
    clReleaseEvent(read_evt);

    free(h_all_habitats);
    free(h_all_flows);
    free(h_counts);
    free(overlap_matrix);

    free_file_list(&habi_list);
    free_file_list(&flow_list);

    free_raster_list(habi_raster, habi_list.count);
    habi_raster = NULL;

    free_raster_list(flow_raster, flow_list.count);
    flow_raster = NULL;

    free(kernel_name);

    clReleaseMemObject(d_all_habitats);
    clReleaseMemObject(d_all_flows);
    clReleaseMemObject(d_counts);

    clReleaseKernel(overlap_k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);

    double ms_total_program = cpu_timer_stop_ms(total_program_timer);

    printf("\n");
    printf("--- TEMPI DEL PROGRAMMA ---\n");
    printf("Tempo pre-processing:   %8.2f ms\n", ms_preprocessing);
    printf("Tempo CPU post-proces:  %8.2f ms\n", ms_total_cpu);
    printf("Tempo totale:           %8.2f ms\n", ms_total_program);

    return 0;
}