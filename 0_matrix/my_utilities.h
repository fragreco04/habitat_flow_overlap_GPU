#ifndef MY_UTILITIES
#define MY_UTILITIES

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>

static inline bool is_file_raw(const char *path) {
    if (path == NULL) {
        return false;
    }

    const char *last_slash = strrchr(path, '/');
    const char *last_backslash = strrchr(path, '\\');
    const char *filename = path;

    if (last_slash && (!last_backslash || last_slash > last_backslash)) {
        filename = last_slash + 1;
    } else if (last_backslash) {
        filename = last_backslash + 1;
    }

    size_t len = strlen(filename);
    const char *ext = ".raw";
    size_t ext_len = strlen(ext);

    if (len <= ext_len) {
        return false;
    }

    return strcmp(filename + len - ext_len, ext) == 0;
}

static inline bool is_file_txt(const char *path) {
    if (path == NULL) {
        return false;
    }

    const char *last_slash = strrchr(path, '/');
    const char *last_backslash = strrchr(path, '\\');
    const char *filename = path;

    if (last_slash && (!last_backslash || last_slash > last_backslash)) {
        filename = last_slash + 1;
    } else if (last_backslash) {
        filename = last_backslash + 1;
    }

    size_t len = strlen(filename);
    const char *ext = ".txt";
    size_t ext_len = strlen(ext);

    if (len <= ext_len) {
        return false;
    }

    return strcmp(filename + len - ext_len, ext) == 0;
}

/* Otteniamo informazioni circa i dati che usiamo */

typedef struct {
    float *data;
    size_t total_cells;
    size_t cols;
    size_t rows;
} Raster;

static inline Raster load_data_from_raw(const char *filepath) {
    Raster r = {NULL, 0, 0, 0};

    FILE *f = fopen(filepath, "rb");
    if (!f) {
        perror("Error: fopen\n");
        return r;
    }

    fseek(f, 0, SEEK_END);
    long file_bytes = ftell(f);
    rewind(f);

    if (file_bytes <= 0 || (size_t)file_bytes % sizeof(float) != 0) {
        fprintf(stderr, "Error: file size not valid for float32 (%ld bytes)\n", file_bytes);
        fclose(f);
        return r;
    }

    r.total_cells = (size_t)file_bytes / sizeof(float);
    int side = (int)round(sqrt((double)r.total_cells));

    if ((size_t)side * side == r.total_cells) {
        r.cols = (size_t)side;
        r.rows = (size_t)side;
    } else {
        fprintf(stderr, "Error: matrix is not a square matrix (%s)\n", filepath);
        fclose(f);
        return r;
    }

    r.data = (float *)malloc((size_t)file_bytes);
    if (!r.data) {
        perror("Error: memory allocation\n");
        fclose(f);
        return r;
    }

    size_t read_count = fread(r.data, sizeof(float), r.total_cells, f);
    fclose(f);

    if (read_count != r.total_cells) {
        fprintf(stderr, "Error: reading data from file\n");
        free(r.data);
        r.data = NULL;
        return r;
    }

    return r;
}

static inline Raster load_data_from_txt(const char *filepath) {
    Raster r = {NULL, 0, 0, 0};

    FILE *f = fopen(filepath, "r");
    if (!f) {
        perror("Error: fopen");
        return r;
    }

    size_t capacity = 1024;
    r.data = (float *)malloc(capacity * sizeof(float));
    if (!r.data) {
        perror("Error: memory allocation");
        fclose(f);
        return r;
    }

    char *line = NULL;
    size_t line_len = 0;
    ssize_t nread;

    while ((nread = getline(&line, &line_len, f)) != -1) {
        char *ptr = line;
        char *endptr;
        size_t cols_in_row = 0;

        while (1) {
            float val = strtof(ptr, &endptr);
            if (ptr == endptr) {
                break;
            }
            ptr = endptr;
            cols_in_row++;

            if (r.total_cells >= capacity) {
                capacity *= 2;
                float *temp = (float *)realloc(r.data, capacity * sizeof(float));
                if (!temp) {
                    perror("Error: realloc");
                    free(r.data);
                    free(line);
                    fclose(f);
                    r.data = NULL;
                    return (Raster){NULL, 0, 0, 0};
                }
                r.data = temp;
            }

            r.data[r.total_cells++] = val;
        }

        if (cols_in_row == 0) {
            continue;
        }

        if (r.rows == 0) {
            r.cols = cols_in_row;
        } else if (cols_in_row != r.cols) {
            fprintf(stderr, "Error: row %zu has %zu columns, expected %zu (%s)\n",
                    r.rows + 1, cols_in_row, r.cols, filepath);
            free(r.data);
            free(line);
            fclose(f);
            return (Raster){NULL, 0, 0, 0};
        }

        r.rows++;
    }

    free(line);
    fclose(f);

    if (r.total_cells == 0 || r.cols == 0 || r.rows == 0) {
        fprintf(stderr, "Error: empty or invalid matrix file (%s)\n", filepath);
        free(r.data);
        return (Raster){NULL, 0, 0, 0};
    }

    float *shrink = (float *)realloc(r.data, r.total_cells * sizeof(float));
    if (shrink) {
        r.data = shrink;
    }

    return r;
}

// La più grande potenza di 2 minore o uguale a x
static inline size_t floor_pow2(size_t x) {
    if (x == 0) return 0;

    size_t p = 1;
    while ((p <= (SIZE_MAX >> 1)) && ((p << 1) <= x)) {
        p <<= 1;
    }
    return p;
}

typedef struct {
    size_t lws[2];
    size_t gws[2];
    size_t nwg[2];
    size_t total_nwg;
    size_t useful_threads;
    
    // Metriche
    cl_uint compute_units;
    double grid_efficiency;
} GPU_Conf;

static inline void gpu_auto_conf(const cl_device_id d, const cl_kernel k, const size_t rows, const size_t cols, GPU_Conf *conf) {

    cl_int err;

    err = clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(conf->compute_units), &conf->compute_units, NULL);
    ocl_check(err, "CL_DEVICE_MAX_COMPUTE_UNITS");

    size_t max_wg_items_sizes[3];           // numero massimo di work-item che possono essere allocati per ciascuna dimensione all'interno di un singolo work-group
    err = clGetDeviceInfo(d, CL_DEVICE_MAX_WORK_ITEM_SIZES, sizeof(max_wg_items_sizes), max_wg_items_sizes, NULL);
    ocl_check(err, "CL_DEVICE_MAX_WORK_ITEM_SIZES");

    size_t max_kernel_threads;
    err = clGetKernelWorkGroupInfo(k, d, CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &max_kernel_threads, NULL);
    ocl_check(err, "CL_KERNEL_WORK_GROUP_SIZE");

    size_t preferred_wg_multiple;
    err = clGetKernelWorkGroupInfo(k, d, CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE, sizeof(size_t), &preferred_wg_multiple, NULL);
    ocl_check(err, "CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE");
    if (preferred_wg_multiple == 0) preferred_wg_multiple = 32;

    size_t threads_per_wg = 256;         // 128 o 256 sono uno sweet-spot per l'occupancy sulla maggior parte dei device
    if (threads_per_wg > max_kernel_threads)
        threads_per_wg = max_kernel_threads;

    // Arrotondiamo per difetto
    threads_per_wg = (threads_per_wg / preferred_wg_multiple) * preferred_wg_multiple;

    size_t cand_x = 32;         // Dim candidata per asse-x, ampiezza tipica per completare un warp su una riga
    if (cand_x > threads_per_wg) cand_x = threads_per_wg;
    if (cand_x > max_wg_items_sizes[0]) cand_x = max_wg_items_sizes[0];

    size_t cand_y = threads_per_wg / cand_x;
    if (cand_y > max_wg_items_sizes[1]) cand_y = max_wg_items_sizes[1];

    // Se la matrice ha meno colonne del blocco cand_x, riduciamo cand_x per evitare sprechi
    while (cand_x > 16 && cand_x > cols) {
        cand_x /= 2;
        cand_y = threads_per_wg / cand_x;
    }

    conf->lws[0] = cand_x;
    conf->lws[1] = cand_y;

    conf->gws[0] = round_mul_up(cols, conf->lws[0]);
    conf->gws[1] = round_mul_up(rows, conf->lws[1]);

    conf->nwg[0] = conf->gws[0] / conf->lws[0];
    conf->nwg[1] = conf->gws[1] / conf->lws[1];
    conf->total_nwg = conf->nwg[0] * conf->nwg[1];

    // Efficienza
    size_t allocated_threads = conf->gws[0] * conf->gws[1];
    conf->useful_threads = rows * cols;
    conf->grid_efficiency = (allocated_threads > 0)
                            ? ((double)conf->useful_threads / (double)allocated_threads)
                            : 0.0;
}

static inline void verify(const cl_int *result, size_t cols, size_t rows) {

    size_t total_elements = (size_t)rows * cols;
    size_t h = 0;
    size_t overlap = 0;

    for (size_t idx = 0; idx < total_elements; ++idx) {
        cl_int val = result[idx];

        if (val & 1) {      // Verifica se dispari
            h++;
        }
        if (val == 3) {
            overlap++;
        }
    }

    float overlap_ratio = (h > 0) ? ((float)overlap / (float)h) : 0.0f;

    printf("\n");
    printf("Habitat and Flow active cells: %zu\n", overlap);
    printf("Habitat active cells: %zu\n", h);

    printf("Result: %.2f\n", overlap_ratio);
}

typedef struct timespec CpuTimer;

static inline CpuTimer cpu_timer_start(void) {
    CpuTimer t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return t;
}

// Restituisce il tempo trascorso in millisecondi (ms) con precisione decimale
static inline double cpu_timer_stop_ms(CpuTimer start) {
    CpuTimer end;
    clock_gettime(CLOCK_MONOTONIC, &end);
    
    double sec  = (double)(end.tv_sec - start.tv_sec);
    double nsec = (double)(end.tv_nsec - start.tv_nsec);
    
    return (sec * 1000.0) + (nsec / 1.0e6);
}

#endif