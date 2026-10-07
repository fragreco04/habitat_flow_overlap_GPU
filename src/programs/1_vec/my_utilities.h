#ifndef MY_UTILITIES
#define MY_UTILITIES

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <math.h>
#include <time.h>

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
    size_t width;
    size_t height;
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
        r.width = (size_t)side;
        r.height = (size_t)side;
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

            if (r.height == 0) {
                r.width = cols_in_row;
            } else if (cols_in_row != r.width) {
                fprintf(stderr, "Error: row %zu has %zu columns, expected %zu (%s)\n",
                        r.height + 1, cols_in_row, r.width, filepath);
                free(r.data);
                free(line);
                fclose(f);
                return (Raster){NULL, 0, 0, 0};
            }

            r.height++;
        }

        free(line);
        fclose(f);

        if (r.total_cells == 0 || r.width == 0 || r.height == 0) {
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
    size_t lws;
    size_t gws;
    size_t total_nwg;
    size_t useful_threads;
    
    // Metriche
    cl_uint compute_units;
    double grid_efficiency;
} GPU_Conf;

static inline void gpu_auto_conf(const cl_device_id d, const cl_kernel k, const size_t nels, const size_t vec, GPU_Conf *conf) {

    cl_int err;

    err = clGetDeviceInfo(d, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(conf->compute_units), &conf->compute_units, NULL);
    ocl_check(err, "CL_DEVICE_MAX_COMPUTE_UNITS");

    size_t max_wg_items_sizes[3];           // numero massimo di work-item che possono essere allocati per dimensione
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

    if (threads_per_wg > max_wg_items_sizes[0])
        threads_per_wg = max_wg_items_sizes[0];

    // Arrotondiamo per difetto al multiplo del sub-group (warp/wavefront)
    threads_per_wg = (threads_per_wg / preferred_wg_multiple) * preferred_wg_multiple;
    if (threads_per_wg == 0) threads_per_wg = preferred_wg_multiple;

    conf->lws = threads_per_wg;

    conf->useful_threads = round_div_up(nels, vec);

    conf->gws = round_mul_up(conf->useful_threads, conf->lws);

    conf->total_nwg = conf->gws / conf->lws;

    size_t allocated_threads = conf->gws;
    conf->grid_efficiency = (allocated_threads > 0)
                            ? ((double)conf->useful_threads / (double)allocated_threads)
                            : 0.0;
}

static inline void verify(const char *result, size_t cols, size_t rows) {
    size_t total_elements = rows * cols;
    size_t h = 0;
    size_t overlap = 0;

    // 1. Calcolo statistiche
    for (size_t idx = 0; idx < total_elements; ++idx) {
        int val = result[idx];

        if (val & 1) {       // Bit 0 attivo: habitat presente (1 o 3)
            h++;
        }
        if (val == 3) {      // Entrambi attivi (11 in binario)
            overlap++;
        }
    }

    float overlap_ratio = (h > 0) ? ((float)overlap / (float)h) : 0.0f;

    printf("\n--- Risultati Verifica ---\n");
    printf("Habitat and Flow active cells: %zu\n", overlap);
    printf("Habitat active cells: %zu\n", h);
    printf("Result (overlap ratio): %.2f\n", overlap_ratio);

    FILE *fp = fopen("out.txt", "w");
    if (fp == NULL) {
        perror("Errore durante l'apertura del file out.txt");
        return;
    }

    fprintf(fp, "# Statistiche\n");
    fprintf(fp, "Dimensioni: %zu x %zu\n", cols, rows);
    fprintf(fp, "Habitat and Flow active cells: %zu\n", overlap);
    fprintf(fp, "Habitat active cells: %zu\n", h);
    fprintf(fp, "Result: %.4f\n\n", overlap_ratio);

    fprintf(fp, "# Matrice Risultato\n");
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            size_t idx = r * cols + c;
            fprintf(fp, "%u ", (unsigned int)result[idx]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    printf("Matrice e statistiche salvate correttamente in 'out.txt'\n");
}

int check_vec(const int v) {
    if (v == 1) return 1;
    else if (v == 2) return 1;
    else if (v == 4) return 1;
    else if (v == 8) return 1;
    else if (v == 16) return 1;
    else return 0;
}

char *get_kernel_name(const int vec) {
    int len = snprintf(NULL, 0, "overlap_vec%d_k", vec);
    if (len < 0) {
        return NULL;
    }

    char *name = (char *)malloc((size_t)len + 1);
    if (name == NULL) {
        perror("Errore malloc in get_kernel_name");
        return NULL;
    }

    snprintf(name, (size_t)len + 1, "overlap_vec%d_k", vec);

    return name;
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