#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "ocl_boiler.h"
#include "my_utilities.h"
#include "../info.h"

cl_event overlap(
    cl_command_queue que,
    cl_kernel kernel,
    cl_mem habitat,
    cl_mem flow,
    cl_mem out,
    cl_int nels,
    size_t *lws,
    size_t *gws,
    cl_uint num_evt,
    cl_event *event_wait_list
) {
    cl_int err;
    cl_event overlap_evt;

    cl_uint arg_index = 0;

    err = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &habitat);
    ocl_check(err, "init_k set kernel arg %d", arg_index);
    ++arg_index;

    err = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &flow);
    ocl_check(err, "init_k set kernel arg %d", arg_index);
    ++arg_index;

    err = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &out);
    ocl_check(err, "init_k set kernel arg %d", arg_index);
    ++arg_index;

    err = clSetKernelArg(kernel, arg_index, sizeof(cl_int), &nels);
    ocl_check(err, "init_k set kernel arg %d", arg_index);
    ++arg_index;


    err = clEnqueueNDRangeKernel(
        que, kernel,
        1, NULL,
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

    char *kernel_name = get_kernel_name(vec);
    cl_kernel overlap_k = clCreateKernel(prog, kernel_name, &err);
    ocl_check(err, "clCreateKernel %s fallito", kernel_name);

    const size_t nels = habi_raster[0]->total_cells;
    size_t nels_pad = nels;
    if (nels % vec != 0) {
        nels_pad = nels + (vec - (nels % vec));
    }

    cl_int num_vectors = (cl_int)(nels_pad / vec);

    size_t memsize_host = sizeof(float) * nels;
    size_t memsize_dev  = sizeof(float) * nels_pad;
    
    size_t memsize_out_host = sizeof(u_char) * nels;
    size_t memsize_out_dev  = sizeof(u_char) * nels_pad;

    GPU_Conf conf;
    gpu_auto_conf(d, overlap_k, nels, vec, &conf);

    cl_mem d_habitat = clCreateBuffer(ctx, CL_MEM_READ_ONLY, memsize_dev, NULL, &err);
    ocl_check(err, "clCreateBuffer failed (d_habitat)");
    cl_mem d_flow = clCreateBuffer(ctx, CL_MEM_READ_ONLY, memsize_dev, NULL, &err);
    ocl_check(err, "clCreateBuffer failed (d_flow)");

    // Riempiamo il padding con degli zeri
    if (memsize_dev > memsize_host) {
        float zero = 0.0f;
        size_t pad_offset = memsize_host;
        size_t pad_size = memsize_dev - memsize_host;

        err = clEnqueueFillBuffer(que, d_habitat, &zero, sizeof(float), pad_offset, pad_size, 0, NULL, NULL);
        ocl_check(err, "fill padding d_habitat");

        err = clEnqueueFillBuffer(que, d_flow, &zero, sizeof(float), pad_offset, pad_size, 0, NULL, NULL);
        ocl_check(err, "fill padding d_flow");
    }
    
    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, memsize_out_dev, NULL, &err);
    ocl_check(err, "clCreateBuffer failed (d_out)");

    size_t n = habi_list.count;
    size_t m = flow_list.count;

    size_t cols = habi_raster[0]->cols;
    size_t rows = habi_raster[0]->rows;

    u_char *h_out = calloc(nels, sizeof(u_char));
    if (h_out == NULL) { fprintf(stderr, "Error: h_out allocation"); exit(12); }

    float *overlap_matrix = malloc(sizeof(float)*n*m);
    if (overlap_matrix == NULL) { fprintf(stderr, "Error: overlap_matrix allocation"); exit(12); }

    cl_event hab_write_evt = NULL;
    cl_event flow_write_evt = NULL;
    cl_event kernel_evt = NULL;
    cl_event read_evt = NULL;

    cl_ulong total_hab_write_ns = 0;
    cl_ulong total_flow_write_ns = 0;
    cl_ulong total_kernel_ns = 0;
    cl_ulong total_read_ns = 0;

    double ms_total_cpu = 0.0;

    for (size_t i = 0; i < n; ++i) {
        err = clEnqueueWriteBuffer(que, d_habitat, CL_FALSE, 0, memsize_host, habi_raster[i]->data, 0, NULL, &hab_write_evt);
        ocl_check(err, "write habitat [%zu]", i);

        for (size_t j = 0; j < m; ++j) {
            err = clEnqueueWriteBuffer(que, d_flow, CL_FALSE, 0, memsize_host, flow_raster[j]->data, 0, NULL, &flow_write_evt);
            ocl_check(err, "write flow [%zu][%zu]", i, j);

            cl_event kern_waits[] = { hab_write_evt, flow_write_evt };
            kernel_evt = overlap(que, overlap_k, d_habitat, d_flow, d_out, num_vectors, &conf.lws, &conf.gws, 2, kern_waits);

            err = clEnqueueReadBuffer(que, d_out, CL_TRUE, 0, memsize_out_host, h_out, 1, &kernel_evt, &read_evt);
            ocl_check(err, "read out [%zu][%zu]", i, j);

            total_flow_write_ns += runtime_ns(flow_write_evt);
            total_kernel_ns += runtime_ns(kernel_evt);
            total_read_ns += runtime_ns(read_evt);

            clReleaseEvent(flow_write_evt);
            clReleaseEvent(kernel_evt);
            clReleaseEvent(read_evt);

            CpuTimer t_cpu_step = cpu_timer_start();
            overlap_matrix[i * m + j] = get_overlap_value(h_out, cols, rows);
            ms_total_cpu += cpu_timer_stop_ms(t_cpu_step);
        }

        total_hab_write_ns += runtime_ns(hab_write_evt);
        clReleaseEvent(hab_write_evt);
    }

    clFinish(que);

    printf("\n");
    print_matrix(overlap_matrix, n, m);
    printf("\n");

    cl_ulong total_write_ns = total_hab_write_ns + total_flow_write_ns;

    // Byte transitati sul bus pcie (n scritture habitat + n*m scritture flow)
    double bytes_h2d_pcie = ((double)n * (double)memsize_host) + ((double)(n * m) * (double)memsize_host);
    double bytes_d2h_pcie = (double)(memsize_out_host * n * m);

    // Byte elaborati dalla DRAM della GPU (senza contare il padding)
    double total_kernel_read_bytes = (double)(2 * memsize_host * n * m);
    double total_kernel_write_bytes = (double)(memsize_out_host * n * m);

    double effective_bw = ((total_kernel_read_bytes + total_kernel_write_bytes) / (total_kernel_ns * 1.0e-9)) / 1.0e9;
    double bw_efficiency = (peak_memory_bandwidth > 0.0) 
                           ? (effective_bw / peak_memory_bandwidth) * 100.0 
                           : 0.0;

    printf("\n");
    printf("==================== REPORT PRESTAZIONALE [Vec: %zu] ====================\n", vec);
    printf("Configurazione Griglia: GWS [%zu] | LWS [%zu]\n", 
           conf.gws, conf.lws);
    printf("Grid Efficiency:       %8.2f %% (Thread utili: %zu / Totali: %zu)\n", 
           (double)(conf.grid_efficiency * 100), conf.useful_threads, conf.gws);
    printf("--------------------------------------------------------------\n");
    printf("Trasferimento H2D:     %8.2f ms | Banda PCIe: %6.2f GB/s\n", 
           total_write_ns * 1.0e-6, bytes_h2d_pcie / (double)total_write_ns);
    printf("Esecuzione Kernel:     %8.2f ms\n", 
           total_kernel_ns * 1.0e-6);
    printf("Trasferimento D2H:     %8.2f ms | Banda PCIe: %6.2f GB/s\n", 
           total_read_ns * 1.0e-6, bytes_d2h_pcie / (double)total_read_ns);
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

        double json_kernel = (double)total_kernel_ns * 1.0e-6;
        double json_h2d = (double)total_write_ns * 1.0e-6;
        double json_d2h = (double)total_read_ns * 1.0e-6;
        double json_cpu_post = ms_total_cpu;
        double json_effective_bw = effective_bw;
        double json_useful_threads = (double)conf.useful_threads * (double)(n * m);
    
        int version = 2;
    
        if (is_warm_up != 1) {
            if (append_benchmark_to_json(json_filepath, version, vec, json_kernel, json_h2d, json_d2h, json_cpu_post, json_effective_bw, json_useful_threads) == 0) {
            printf("\nBenchmark salvato con successo in %s [versione %d, vec %zu]\n", json_filepath, version, vec);
            } else {
            fprintf(stderr, "\nErrore durante il salvataggio in %s\n", json_filepath);
            }
        }
    }

    // -----------------------------------------------------------


    free(h_out);
    free(overlap_matrix);

    free_file_list(&habi_list);
    free_file_list(&flow_list);

    free_raster_list(habi_raster, habi_list.count);
    habi_raster = NULL;

    free_raster_list(flow_raster, flow_list.count);
    flow_raster = NULL;

    free(kernel_name);

    clReleaseMemObject(d_habitat);
    clReleaseMemObject(d_flow);
    clReleaseMemObject(d_out);

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