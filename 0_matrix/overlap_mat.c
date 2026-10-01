#include <stdio.h>
#include <stdlib.h>

#include "ocl_boiler.h"
#include "my_utilities.h"
#include "../info.h"

 cl_event overlap(
    cl_command_queue que,
    cl_kernel kernel,
    cl_mem habitat,
    cl_mem flow,
    cl_mem result,
    cl_int cols,
    cl_int rows,
    size_t *lws,
    size_t *gws,
    cl_uint num_events_in_wait_list,
    cl_event *event_list
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

    err = clSetKernelArg(kernel, arg_index, sizeof(cl_mem), &result);
    ocl_check(err, "init_k set kernel arg %d", arg_index);
    ++arg_index;

    err = clSetKernelArg(kernel, arg_index, sizeof(cl_int), &cols);
    ocl_check(err, "init_k set kernel arg %d", arg_index);
    ++arg_index;

    err = clSetKernelArg(kernel, arg_index, sizeof(cl_int), &rows);
    ocl_check(err, "init_k set kernel arg %d", arg_index);
    ++arg_index;


    err = clEnqueueNDRangeKernel(
        que, kernel,
        2, NULL,
        gws, lws,
        num_events_in_wait_list, event_list,
        &overlap_evt
    );

    ocl_check(err, "enqueue kernel overlap_k");
    return overlap_evt;
}

int main(int argc, char* argv[]) {

    CpuTimer total_program_timer = cpu_timer_start();
    CpuTimer pre_processing_timer = cpu_timer_start();
    
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <habitat_filepath> <flow_filepath>\n", argv[0]);
        exit(1);
    }

    // Controlliamo se i file in input sono validi
    const char *habi_filepath = argv[1];
    if (!is_file_txt(habi_filepath)) {
        fprintf(stderr, "Habitat data file not valid\n");
        exit(2);
    }

    const char *flow_filepath = argv[2];
    if (!is_file_txt(flow_filepath)) {
        fprintf(stderr, "Flow data file not valid\n");
        exit(3);
    }

    // Controlliamo se i file contengono dati validi
    Raster habitat = load_data_from_txt(habi_filepath);
    if (habitat.data == NULL) {
        fprintf(stderr, "Error: invalid habitat data pointer\n");
        exit(5);
    }

    Raster flow = load_data_from_txt(flow_filepath);
    if (flow.data == NULL) {
        fprintf(stderr, "Error: invalid flow data pointer\n");
        exit(6);
    }

    // Controlliamo che le due matrici hanno le stesse dimensioni
    if (habitat.cols != flow.cols) {
        fprintf(stderr, "the two matrices have different widths\n");
        exit(7);
    }

    if (habitat.rows != flow.rows) {
        fprintf(stderr, "the two matrices have different heights\n");
        exit(8);
    }

    cl_platform_id p = select_platform();
    cl_device_id d = select_device(p);
    cl_context ctx = create_context(p, d);
    cl_command_queue que = create_queue(ctx, d);
    printf("\n");

    double ms_preprocessing = cpu_timer_stop_ms(pre_processing_timer);

    const size_t cols = habitat.cols;
    const size_t rows = habitat.rows;
    const size_t nels = cols * rows;

    /***  Inizio del codice per la GPU ***/

    cl_program prog = create_program("overlap.ocl", ctx, d);

    cl_int err;

    // Creiamo il kernel
    char *kernel_name = "overlap_k";

    cl_kernel overlap_k = clCreateKernel(prog, kernel_name, &err);
    ocl_check(err, "clCreateKernel %s fallito", kernel_name);
    
    GPU_Conf conf;
    gpu_auto_conf(d, overlap_k, cols, rows, &conf);

    size_t memsize = sizeof(float)*nels;
    size_t memsize_out = sizeof(cl_int)*nels;

    // Creiamo i buffer con le matrici
    cl_mem d_habitat = clCreateBuffer(ctx, CL_MEM_READ_ONLY, memsize, NULL, &err);
    ocl_check(err, "clCreateBuffer failed (d_habitat)");
    cl_mem d_flow = clCreateBuffer(ctx, CL_MEM_READ_ONLY, memsize, NULL, &err);
    ocl_check(err, "clCreateBuffer failed (d_flow)");

    cl_mem d_out = clCreateBuffer(ctx, CL_MEM_WRITE_ONLY | CL_MEM_ALLOC_HOST_PTR, memsize_out, NULL, &err);
    ocl_check(err, "clCreateBuffer failed (d_out)");

    // Riempiamo i buffer
    cl_event write_wait_list[2];

    err = clEnqueueWriteBuffer(que, d_habitat, CL_FALSE, 0, memsize, habitat.data, 0, NULL, &write_wait_list[0]);
    err = clEnqueueWriteBuffer(que, d_flow, CL_FALSE, 0, memsize, flow.data, 0, NULL, &write_wait_list[1]);

    cl_event kernel_evt;
    kernel_evt = overlap(que, overlap_k, d_habitat, d_flow, d_out, (cl_int)cols, (cl_int)rows, conf.lws, conf.gws, 2, write_wait_list);

    cl_event wait_list[1] = { kernel_evt };
    cl_event read_evt;

    int *h_result = calloc(nels, sizeof(int));

    err = clEnqueueReadBuffer(que, d_out, CL_FALSE, 0, memsize_out, h_result, 1, wait_list, &read_evt);
    ocl_check(err, "read buffer d_out");

    clWaitForEvents(1, &read_evt);

    CpuTimer verify_timer = cpu_timer_start();
    verify(h_result, habitat.cols, habitat.rows);
    double ms_verify = cpu_timer_stop_ms(verify_timer);

    cl_ulong write_ns = runtime_ns(write_wait_list[0]) + runtime_ns(write_wait_list[1]);
    cl_ulong kernel_ns = runtime_ns(kernel_evt);
    cl_ulong read_ns = runtime_ns(read_evt);

    double total_read_bytes = memsize*2;
    double total_write_bytes = memsize_out;

    double effective_bw = ((total_read_bytes + total_write_bytes) / (kernel_ns * 1.0e-9)) / 1.0e9;
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
           write_ns * 1.0e-6, total_read_bytes / (double)write_ns);
    printf("Esecuzione Kernel:     %8.2f ms\n", 
           kernel_ns * 1.0e-6);
    printf("Trasferimento D2H:     %8.2f ms | Banda PCIe: %6.2f GB/s\n", 
           read_ns * 1.0e-6, total_write_bytes / (double)read_ns);
    printf("--------------------------------------------------------------\n");
    printf("Banda DRAM Effettiva:  %8.2f GB/s\n", effective_bw);
    printf("Banda DRAM Teorica:    %8.2f GB/s\n", peak_memory_bandwidth);
    printf("Efficienza di Memoria: %8.2f %%\n", bw_efficiency);
    printf("==============================================================\n\n");

    clFinish(que);

    clReleaseEvent(write_wait_list[0]);
    clReleaseEvent(write_wait_list[1]);
    clReleaseEvent(kernel_evt);
    clReleaseEvent(read_evt);

    clReleaseMemObject(d_habitat);
    clReleaseMemObject(d_flow);
    clReleaseMemObject(d_out);

    clReleaseKernel(overlap_k);
    clReleaseProgram(prog);
    clReleaseCommandQueue(que);
    clReleaseContext(ctx);

    free(habitat.data);
    free(flow.data);

    double ms_total_program = cpu_timer_stop_ms(total_program_timer);

    printf("\n");
    printf("--- TEMPI DEL PROGRAMMA ---\n");
    printf("Tempo pre-processing:   %8.2f ms\n", ms_preprocessing);
    printf("Tempo verifica riusult: %8.2f ms\n", ms_verify);
    printf("Tempo totale:           %8.2f ms\n", ms_total_program);

    return 0;
}