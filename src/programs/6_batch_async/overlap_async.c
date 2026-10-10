#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "ocl_boiler.h"
#include "my_utilities.h"
#include "../../modules/info.h"

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

    if (argc < 6 || argc > 8) {
        fprintf(stderr, "Usage: %s <habitat_filepath> <flow_filepath> <vec> <warm_up> <json> [n_batch] [m_batch]\n", argv[0]);
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

    if (habi_list.paths == NULL || flow_list.paths == NULL || habi_list.count == 0 || flow_list.count == 0) {
        fprintf(stderr, "Error: missing or empty directories\n");
        free_file_list(&habi_list);
        free_file_list(&flow_list);
        exit(5);
    }

    printf("\nHabitat files:\n");
    for (size_t i = 0; i < habi_list.count; ++i) printf("\t%s\n", habi_list.paths[i]);
    printf("\nFlow files:\n");
    for (size_t i = 0; i < flow_list.count; ++i) printf("\t%s\n", flow_list.paths[i]);
    printf("\nHabitat: %zu - Flow: %zu\n", habi_list.count, flow_list.count);

    Raster **habi_raster = calloc(habi_list.count, sizeof(Raster *));
    Raster **flow_raster = calloc(flow_list.count, sizeof(Raster *));

    for (size_t i = 0; i < habi_list.count; ++i) habi_raster[i] = load_data_from_txt(habi_list.paths[i]);
    for (size_t i = 0; i < flow_list.count; ++i) flow_raster[i] = load_data_from_txt(flow_list.paths[i]);

    if (!check_same_dimensions(habi_raster, habi_list.count, flow_raster, flow_list.count)) {
        fprintf(stderr, "Matrix must have same sizes\n");
        exit(10);
    }

    size_t n = habi_list.count;
    size_t m = flow_list.count;

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue queues[2];
    queues[0] = create_queue(ctx, d);
    queues[1] = create_queue(ctx, d);
    printf("\n");

    cl_int err;
    cl_program prog = create_program("overlap.ocl", ctx, d);
    char *kernel_name = get_kernel_name(vec);
    
    // Creiamo due istanze separate del kernel per evitare race condition
    cl_kernel overlap_k[2];
    overlap_k[0] = clCreateKernel(prog, kernel_name, &err);
    ocl_check(err, "clCreateKernel 0 fallito");
    overlap_k[1] = clCreateKernel(prog, kernel_name, &err);
    ocl_check(err, "clCreateKernel 1 fallito");

    const size_t nels = habi_raster[0]->total_cells;
    size_t nels_pad = nels;
    if (nels % vec != 0) 
        nels_pad = nels + (vec - (nels % vec));

    cl_uint num_vectors = (cl_uint)(nels_pad / vec);
    size_t memsize_host = sizeof(u_char) * nels;
    size_t memsize_dev  = sizeof(u_char) * nels_pad;

    // Autotuning della Memoria
    size_t *batch_values = mem_auto_conf(argc, argv, d, n, m, memsize_dev);
    if (batch_values == NULL) {
        fprintf(stderr, "Error: mem_auto_conf");
        exit(11);
    }

    size_t n_batch = batch_values[0];
    size_t m_batch = batch_values[1];
    free(batch_values);

    // Pre-calcoliamo la configurazione base
    GPU_Conf base_conf;
    gpu_auto_conf(d, overlap_k[0], nels, vec, 1, &base_conf);

    double ms_preprocessing = cpu_timer_stop_ms(pre_processing_timer);

    // Memoria Host Pinned (Solo per Matrici Massicce)
    cl_mem p_hab = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, n * memsize_dev, NULL, &err);
    u_char *h_all_habitats = clEnqueueMapBuffer(queues[0], p_hab, CL_TRUE, CL_MAP_WRITE, 0, n * memsize_dev, 0, NULL, NULL, &err);

    cl_mem p_flow = clCreateBuffer(ctx, CL_MEM_READ_ONLY | CL_MEM_ALLOC_HOST_PTR, m * memsize_dev, NULL, &err);
    u_char *h_all_flows = clEnqueueMapBuffer(queues[0], p_flow, CL_TRUE, CL_MAP_WRITE, 0, m * memsize_dev, 0, NULL, NULL, &err);

    for (size_t i = 0; i < n; ++i) 
        memcpy(h_all_habitats + (i * nels_pad), habi_raster[i]->data, memsize_host);
    for (size_t j = 0; j < m; ++j) 
        memcpy(h_all_flows + (j * nels_pad), flow_raster[j]->data, memsize_host);

    // FIX INTEL DRIVER: Sostituzione dei buffer mappati con malloc standard per gli array minuscoli di output
    cl_uint *h_counts[2];
    for (int i = 0; i < 2; ++i) {
        h_counts[i] = malloc(n_batch * m_batch * 2 * sizeof(cl_uint));
        if (!h_counts[i]) { fprintf(stderr, "Error: h_counts allocation\n"); exit(12); }
    }

    // Memoria Device Double Buffered
    cl_mem d_habitats = clCreateBuffer(ctx, CL_MEM_READ_ONLY, n_batch * memsize_dev, NULL, &err);
    cl_mem d_flows[2];  
    cl_mem d_counts[2];
    for (int i = 0; i < 2; ++i) {
        d_flows[i] = clCreateBuffer(ctx, CL_MEM_READ_ONLY, m_batch * memsize_dev, NULL, &err);
        d_counts[i] = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY, n_batch * m_batch * 2 * sizeof(cl_uint), NULL, &err);
    }

    float *overlap_matrix = malloc(sizeof(float) * n * m);

    // Variabili Tracciamento Pipeline
    cl_event hab_write_evt = NULL;
    cl_event flow_write_evts[2] = {NULL, NULL};
    cl_event fill_evts[2] = {NULL, NULL};
    cl_event kernel_evts[2] = {NULL, NULL};
    cl_event read_evts[2] = {NULL, NULL};

    bool has_data[2] = {false, false};
    size_t track_h[2], track_f[2], track_n[2], track_m[2];

    cl_ulong total_write_hab_ns = 0, total_write_flow_ns = 0;
    cl_ulong total_kernel_ns = 0, total_read_ns = 0;
    double ms_total_cpu = 0.0;
    
    GPU_Conf conf_final;

    // Esecuzione Tiling 2D Pipelined
    size_t h_idx = 0;
    while (h_idx < n) {
        size_t curr_n = (n - h_idx < n_batch) ? (n - h_idx) : n_batch;
        
        err = clEnqueueWriteBuffer(queues[0], d_habitats, CL_FALSE, 0, curr_n * memsize_dev, 
                                  h_all_habitats + (h_idx * nels_pad), 0, NULL, &hab_write_evt);
        ocl_check(err, "write habitats chunk");

        size_t f_idx = 0;
        int step = 0;
        
        while (f_idx < m) {
            size_t curr_m = (m - f_idx < m_batch) ? (m - f_idx) : m_batch;
            int q = step % 2;

            if (has_data[q]) {
                clWaitForEvents(1, &read_evts[q]);
                
                total_write_flow_ns += runtime_ns(flow_write_evts[q]);
                total_kernel_ns     += runtime_ns(kernel_evts[q]);
                total_read_ns       += runtime_ns(read_evts[q]);

                CpuTimer cpu_timer = cpu_timer_start();
                for (size_t local_h = 0; local_h < track_n[q]; ++local_h) {
                    for (size_t local_f = 0; local_f < track_m[q]; ++local_f) {
                        size_t global_h = track_h[q] + local_h;
                        size_t global_f = track_f[q] + local_f;
                        size_t pair_idx_local = local_h * track_m[q] + local_f;
                        size_t pair_idx_global = global_h * m + global_f;

                        cl_uint h_val = h_counts[q][pair_idx_local * 2];
                        cl_uint overlap_val = h_counts[q][pair_idx_local * 2 + 1];
                        overlap_matrix[pair_idx_global] = (h_val > 0) ? ((float)overlap_val / (float)h_val) : 0.0f;
                    }
                }
                ms_total_cpu += cpu_timer_stop_ms(cpu_timer);

                clReleaseEvent(flow_write_evts[q]);
                clReleaseEvent(fill_evts[q]);
                clReleaseEvent(kernel_evts[q]);
                clReleaseEvent(read_evts[q]);
                has_data[q] = false;
            }

            track_h[q] = h_idx;
            track_f[q] = f_idx;
            track_n[q] = curr_n;
            track_m[q] = curr_m;
            has_data[q] = true;

            err = clEnqueueWriteBuffer(queues[q], d_flows[q], CL_FALSE, 0, curr_m * memsize_dev, 
                                      h_all_flows + (f_idx * nels_pad), 0, NULL, &flow_write_evts[q]);
            
            cl_uint zero = 0;
            err = clEnqueueFillBuffer(queues[q], d_counts[q], &zero, sizeof(cl_uint), 0, 
                                      curr_n * curr_m * 2 * sizeof(cl_uint), 0, NULL, &fill_evts[q]);

            GPU_Conf conf_batch = base_conf;
            conf_batch.gws[1] = curr_n * curr_m;
            conf_final = conf_batch; 

            cl_uint num_wait = (q == 1) ? 1 : 0;
            cl_event *wait_ptr = (q == 1) ? &hab_write_evt : NULL;

            kernel_evts[q] = overlap(queues[q], overlap_k[q], d_habitats, d_flows[q], 
                                     num_vectors, (cl_uint)curr_m, d_counts[q], 
                                     conf_batch.lws, conf_batch.gws, num_wait, wait_ptr);

            err = clEnqueueReadBuffer(queues[q], d_counts[q], CL_FALSE, 0, 
                                      curr_n * curr_m * 2 * sizeof(cl_uint), 
                                      h_counts[q], 0, NULL, &read_evts[q]);
            ocl_check(err, "read d_counts chunk");

            f_idx += curr_m;
            step++;
        }

        // Svuotamento della coda di fine blocco
        for (int q = 0; q < 2; ++q) {
            if (has_data[q]) {
                clWaitForEvents(1, &read_evts[q]);
                
                total_write_flow_ns += runtime_ns(flow_write_evts[q]);
                total_kernel_ns     += runtime_ns(kernel_evts[q]);
                total_read_ns       += runtime_ns(read_evts[q]);

                CpuTimer cpu_timer = cpu_timer_start();
                for (size_t local_h = 0; local_h < track_n[q]; ++local_h) {
                    for (size_t local_f = 0; local_f < track_m[q]; ++local_f) {
                        size_t global_h = track_h[q] + local_h;
                        size_t global_f = track_f[q] + local_f;
                        size_t pair_idx_local = local_h * track_m[q] + local_f;
                        size_t pair_idx_global = global_h * m + global_f;

                        cl_uint h_val = h_counts[q][pair_idx_local * 2];
                        cl_uint overlap_val = h_counts[q][pair_idx_local * 2 + 1];
                        overlap_matrix[pair_idx_global] = (h_val > 0) ? ((float)overlap_val / (float)h_val) : 0.0f;
                    }
                }
                ms_total_cpu += cpu_timer_stop_ms(cpu_timer);

                clReleaseEvent(flow_write_evts[q]);
                clReleaseEvent(fill_evts[q]);
                clReleaseEvent(kernel_evts[q]);
                clReleaseEvent(read_evts[q]);
                has_data[q] = false;
            }
        }

        total_write_hab_ns += runtime_ns(hab_write_evt);
        clReleaseEvent(hab_write_evt);
        hab_write_evt = NULL;

        h_idx += curr_n;
    }

    clFinish(queues[0]);
    clFinish(queues[1]);

    printf("\n");
    print_matrix(overlap_matrix, n, m);

    cl_ulong total_write_ns = total_write_hab_ns + total_write_flow_ns;

    double iterations_outer = ceil((double)n / n_batch);
    double bytes_h2d_pcie = (memsize_dev * n) + (memsize_dev * m * iterations_outer);
    double bytes_d2h_pcie = (double)(n * m) * (double)(2 * sizeof(cl_uint));

    double total_kernel_read_bytes = 2.0 * (double)memsize_dev * (double)(n * m);
    double total_kernel_write_bytes = bytes_d2h_pcie;

    double effective_bw = ((total_kernel_read_bytes + total_kernel_write_bytes) / (total_kernel_ns * 1.0e-9)) / 1.0e9;
    double bw_efficiency = (peak_memory_bandwidth > 0.0) ? (effective_bw / peak_memory_bandwidth) * 100.0 : 0.0;

    printf("\n");
    printf("==================== REPORT PRESTAZIONALE ====================\n");
    printf("Configurazione Griglia: GWS [%zu, %zu] | LWS [%zu, %zu]\n", 
           conf_final.gws[0], conf_final.gws[1], conf_final.lws[0], conf_final.lws[1]);
    printf("Grid Efficiency:       %8.2f %% (Thread utili: %zu / Totali: %zu)\n", 
           (double)(conf_final.grid_efficiency * 100), conf_final.useful_threads, conf_final.gws[0]*conf_final.gws[1]);
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

    char *json_filepath = argv[5];
    if (strcmp(json_filepath, "no") == 0) {
        printf("No salvataggio come test.\n");
    } else {
        int is_warm_up = atoi(argv[4]);
        double json_kernel = (double)total_kernel_ns * 1.0e-6;
        double json_h2d = (double)total_write_ns * 1.0e-6;
        double json_d2h = (double)total_read_ns * 1.0e-6;
        double json_effective_bw = effective_bw;
        double json_useful_threads = conf_final.useful_threads;
        int version = 4;
    
        if (is_warm_up != 1) {
            if (append_benchmark_to_json(json_filepath, version, vec, json_kernel, json_h2d, json_d2h, ms_total_cpu, json_effective_bw, json_useful_threads) == 0) {
                printf("Benchmark salvato con successo in %s [versione %d, vec %zu]\n", json_filepath, version, vec);
            } else {
                fprintf(stderr, "Errore durante il salvataggio in %s\n", json_filepath);
            }
        }
    }

    clEnqueueUnmapMemObject(queues[0], p_hab, h_all_habitats, 0, NULL, NULL);
    clEnqueueUnmapMemObject(queues[0], p_flow, h_all_flows, 0, NULL, NULL);

    clReleaseMemObject(p_hab);
    clReleaseMemObject(p_flow);
    clReleaseMemObject(d_habitats);
    for (int i = 0; i < 2; ++i) {
        clReleaseMemObject(d_flows[i]);
        clReleaseMemObject(d_counts[i]);
        free(h_counts[i]);
    }

    free(overlap_matrix);
    free_file_list(&habi_list);
    free_file_list(&flow_list);
    free_raster_list(habi_raster, habi_list.count);
    free_raster_list(flow_raster, flow_list.count);
    free(kernel_name);

    clReleaseKernel(overlap_k[0]);
    clReleaseKernel(overlap_k[1]);
    clReleaseProgram(prog);
    clReleaseCommandQueue(queues[0]);
    clReleaseCommandQueue(queues[1]);
    clReleaseContext(ctx);

    double ms_total_program = cpu_timer_stop_ms(total_program_timer);

    printf("\n--- TEMPI DEL PROGRAMMA ---\n");
    printf("Tempo pre-processing:   %8.2f ms\n", ms_preprocessing);
    printf("Tempo CPU post-proces:  %8.2f ms\n", ms_total_cpu);
    printf("Tempo totale:           %8.2f ms\n", ms_total_program);

    return 0;
}