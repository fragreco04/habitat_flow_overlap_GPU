#ifndef MY_UTILITIES
#define MY_UTILITIES

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <ctype.h>
#include "../cJSON.h"
#include <math.h>

// --------------------------------------------------
//      Controllo sui file
// --------------------------------------------------

typedef struct {
    char **paths;
    size_t count;
} FileList;

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

static inline bool check_dir(const char *dir_path) {
    if (dir_path == NULL) {
        return false;
    }

    DIR *dir = opendir(dir_path);
    if (dir == NULL) {
        return false;
    }

    struct dirent *entry;
    char full_path[1024];

    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        snprintf(full_path, sizeof(full_path), "%s/%s", dir_path, entry->d_name);

        struct stat st;
        if (stat(full_path, &st) != 0) {
            closedir(dir);
            return false;
        }

        if (S_ISREG(st.st_mode)) {
            if (!is_file_txt(entry->d_name)) {
                closedir(dir);
                return false;
            }
        }
    }

    closedir(dir);
    return true;
}

static inline void free_file_list(FileList *list) {
    if (list == NULL || list->paths == NULL) return;
    for (size_t i = 0; i < list->count; i++) {
        free(list->paths[i]);
    }
    free(list->paths);
    list->paths = NULL;
    list->count = 0;
}

static int extract_file_number(const char *filepath) {
    int num = 0;
    const char *p = filepath + strlen(filepath) - 1;
    
    while (p > filepath && !isdigit(*p)) p--;
    
    const char *end = p;
    while (p > filepath && isdigit(*p)) p--;
    
    if (!isdigit(*p)) p++;
    
    if (p <= end) {
        num = atoi(p);
    }
    return num;
}

static int compare_paths(const void *a, const void *b) {
    const char *path_a = *(const char **)a;
    const char *path_b = *(const char **)b;
    
    int num_a = extract_file_number(path_a);
    int num_b = extract_file_number(path_b);
    
    if (num_a != num_b) {
        return num_a - num_b;
    }
    return strcmp(path_a, path_b);
}

static inline FileList get_filepaths_list(const char *dir_path) {
    FileList result = { .paths = NULL, .count = 0 };

    if (dir_path == NULL) {
        return result;
    }

    DIR *dir = opendir(dir_path);
    if (dir == NULL) {
        perror("Errore apertura directory");
        return result;
    }

    size_t capacity = 16;
    char **file_list = malloc(capacity * sizeof(char *));
    if (file_list == NULL) {
        closedir(dir);
        return result;
    }

    size_t dir_len = strlen(dir_path);
    const char *sep = (dir_len > 0 && dir_path[dir_len - 1] == '/') ? "" : "/";

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        if (strcmp(entry->d_name, ".") == 0 || strcmp(entry->d_name, "..") == 0) {
            continue;
        }

        char full_path[1024];
        snprintf(full_path, sizeof(full_path), "%s%s%s", dir_path, sep, entry->d_name);

        struct stat path_stat;
        if (stat(full_path, &path_stat) == 0 && S_ISREG(path_stat.st_mode)) {
            if (result.count + 1 >= capacity) {
                capacity *= 2;
                char **temp = realloc(file_list, capacity * sizeof(char *));
                if (temp == NULL) {
                    result.paths = file_list;
                    free_file_list(&result);
                    closedir(dir);
                    return (FileList){NULL, 0};
                }
                file_list = temp;
            }

            file_list[result.count] = strdup(full_path);
            if (file_list[result.count] == NULL) {
                result.paths = file_list;
                free_file_list(&result);
                closedir(dir);
                return (FileList){NULL, 0};
            }
            result.count++;
        }
    }

    closedir(dir);

    if (result.count > 1) {
        qsort(file_list, result.count, sizeof(char *), compare_paths);
    }

    file_list[result.count] = NULL; 
    result.paths = file_list;

    return result;
}

// --------------------------------------------------

int check_vec(const int v) {
    if (v == 1) return 1;
    else if (v == 2) return 1;
    else if (v == 4) return 1;
    else if (v == 8) return 1;
    else if (v == 16) return 1;
    else return 0;
}

char *get_kernel_name(const int vec) {
    int len = snprintf(NULL, 0, "overlap_reduce_batch_vec%d_k", vec);
    if (len < 0) {
        return NULL;
    }

    char *name = (char *)malloc((size_t)len + 1);
    if (name == NULL) {
        perror("Errore malloc in get_kernel_name");
        return NULL;
    }

    snprintf(name, (size_t)len + 1, "overlap_reduce_batch_vec%d_k", vec);

    return name;
}

// --------------------------------------------------


// --------------------------------------------------
//      Carichiamo i dati dai file
// --------------------------------------------------

typedef struct {
    u_char *data;
    size_t total_cells;
    size_t cols;
    size_t rows;
} Raster;

static inline Raster *load_data_from_txt(const char *filepath) {
    FILE *f = fopen(filepath, "r");
    if (!f) {
        perror("Error: fopen");
        return NULL;
    }

    Raster *r = (Raster *)malloc(sizeof(Raster));
    if (!r) {
        perror("Error: raster allocation");
        fclose(f);
        return NULL;
    }

    r->data = NULL;
    r->cols = 0;
    r->rows = 0;
    r->total_cells = 0;

    size_t capacity = 1024;
    r->data = (u_char *)malloc(capacity * sizeof(u_char));
    if (!r->data) {
        perror("Error: memory allocation");
        free(r);
        fclose(f);
        return NULL;
    }

    char *line = NULL;
    size_t line_len = 0;
    ssize_t nread;

    while ((nread = getline(&line, &line_len, f)) > 0) {
        line[nread] = '\0';

        char *ptr = line;
        char *endptr;
        size_t cols_in_row = 0;

        while (*ptr != '\0') {
            while (*ptr != '\0' && isspace((unsigned char)*ptr)) {
                ptr++;
            }
            if (*ptr == '\0') {
                break; 
            }

            float val = strtof(ptr, &endptr);
            if (ptr == endptr) {
                break;
            }
            ptr = endptr;
            cols_in_row++;

            if (r->total_cells >= capacity) {
                capacity *= 2;
                u_char *temp = (u_char *)realloc(r->data, capacity * sizeof(u_char));
                if (!temp) {
                    perror("Error: realloc");
                    free(r->data);
                    free(r);
                    free(line);
                    fclose(f);
                    return NULL;
                }
                r->data = temp;
            }

            r->data[r->total_cells++] = (val > 0.0f) ? 1 : 0;
        }

        if (cols_in_row == 0) {
            continue;
        }

        if (r->rows == 0) {
            r->cols = cols_in_row;
        } else if (cols_in_row != r->cols) {
            fprintf(stderr, "Error: row %zu has %zu columns, expected %zu (%s)\n",
                    r->rows + 1, cols_in_row, r->cols, filepath);
            free(r->data);
            free(r);
            free(line);
            fclose(f);
            return NULL;
        }

        r->rows++;
    }

    free(line);
    fclose(f);

    if (r->total_cells == 0 || r->cols == 0 || r->rows == 0) {
        fprintf(stderr, "Error: empty or invalid matrix file (%s)\n", filepath);
        free(r->data);
        free(r);
        return NULL;
    }

    u_char *shrink = (u_char *)realloc(r->data, r->total_cells * sizeof(u_char));
    if (shrink) {
        r->data = shrink;
    }

    return r;
}

static inline void free_raster_list(Raster **raster_list, size_t count) {
    if (raster_list == NULL) {
        return;
    }

    for (size_t i = 0; i < count; ++i) {
        if (raster_list[i] != NULL) {
            if (raster_list[i]->data != NULL) {
                free(raster_list[i]->data);
                raster_list[i]->data = NULL;
            }
            free(raster_list[i]);
            raster_list[i] = NULL;
        }
    }

    free(raster_list);
}

int check_same_dimensions(Raster **array_a, size_t count_a, Raster **array_b, size_t count_b) {
    if (count_a == 0 && count_b == 0) {
        return 0;
    }

    Raster *first = NULL;
    if (count_a > 0 && array_a != NULL) {
        first = array_a[0];
    } else if (count_b > 0 && array_b != NULL) {
        first = array_b[0];
    }

    if (first == NULL) {
        return 0;
    }

    size_t target_cols = first->cols;
    size_t target_rows = first->rows;

    if (array_a != NULL) {
        for (size_t i = 0; i < count_a; i++) {
            if (array_a[i] == NULL) {
                return 0;
            }
            if (array_a[i]->cols != target_cols || array_a[i]->rows != target_rows) {
                return 0; 
            }
        }
    }

    if (array_b != NULL) {
        for (size_t i = 0; i < count_b; i++) {
            if (array_b[i] == NULL) {
                return 0;
            }
            if (array_b[i]->cols != target_cols || array_b[i]->rows != target_rows) {
                return 0; 
            }
        }
    }

    return 1;
}

// --------------------------------------------------
//      Funzioni sulla GPU
// --------------------------------------------------

typedef struct {
    size_t lws[2];
    size_t gws[2];
    size_t total_nwg;
    size_t useful_threads;
    
    // Metriche
    cl_uint compute_units;
    double grid_efficiency;
} GPU_Conf;

static inline void gpu_auto_conf(const cl_device_id d, const cl_kernel k, const size_t nels, const size_t vec, const size_t total_pairs, GPU_Conf *conf) {

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

    // Configurazione LWS (2D)
    conf->lws[0] = threads_per_wg;
    conf->lws[1] = 1;

    // Thread utili per la dimensione x (vettorizzata)
    size_t useful_x = round_div_up(nels, vec);

    // Configurazione GWS (2D)
    conf->gws[0] = round_mul_up(useful_x, conf->lws[0]);
    conf->gws[1] = total_pairs;

    // Totale complessivo dei thread utili su tutta la griglia 2D
    conf->useful_threads = useful_x * total_pairs;

    // Calcolo work-groups totali
    conf->total_nwg = (conf->gws[0] / conf->lws[0]) * (conf->gws[1] / conf->lws[1]);

    // Grid efficiency
    size_t total_allocated_threads = conf->gws[0] * conf->gws[1];
    conf->grid_efficiency = (total_allocated_threads > 0)
                            ? ((double)conf->useful_threads / (double)total_allocated_threads)
                            : 0.0;
}

// ----------------------------------------------------------------------------------

static inline float get_overlap_value(const u_char *result, size_t cols, size_t rows) {
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

    return overlap_ratio;
}

static inline void print_matrix(const float *data, size_t rows, size_t cols) {
    if (data == NULL || rows == 0 || cols == 0) {
        printf("[Matrice vuota o non valida]\n");
        return;
    }

    printf("Overlap Matrix:\n");
    for (size_t r = 0; r < rows; ++r) {
        for (size_t c = 0; c < cols; ++c) {
            size_t idx = r * cols + c;

            printf("%7.4f ", data[idx]);
        }
        printf("\n");
    }
    printf("\n");
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

// -------------------------------------------------------
// Salvataggio dati in json file
// -------------------------------------------------------


static inline double round_to_3_dec(double val) {
    return round(val * 1000.0) / 1000.0;
}

// Per lo speed-up
int append_benchmark_to_json(
    const char *filepath,
    int version,
    size_t vec,
    double kernel_ms,
    double h2d_ms,
    double d2h_ms,
    double cpu_post_ms,
    double effective_bw,
    double useful_threads
) {
    cJSON *root = NULL;
    char *file_data = NULL;
    long file_len = 0;

    FILE *fp = fopen(filepath, "rb");
    if (fp) {
        fseek(fp, 0, SEEK_END);
        file_len = ftell(fp);
        fseek(fp, 0, SEEK_SET);

        if (file_len > 0) {
            file_data = malloc(file_len + 1);
            if (file_data) {
                size_t bytes_read = fread(file_data, 1, file_len, fp);
                file_data[bytes_read] = '\0';
                root = cJSON_Parse(file_data);
            }
            free(file_data);
        }
        fclose(fp);
    }

    if (!root) {
        root = cJSON_CreateObject();
        if (!root) return -1;
    }

    char version_key[64];
    if (version == 1)
        snprintf(version_key, sizeof(version_key), "1_cpu_baseline");
    else if (version == 2)
        snprintf(version_key, sizeof(version_key), "2_gpu_naive");
    else if (version == 3)
        snprintf(version_key, sizeof(version_key), "3_gpu_reduce");
    else if (version == 4)
        snprintf(version_key, sizeof(version_key), "4_gpu_reduce_batch");
    else {
        fprintf(stderr, "nome lista json sbagliato, lato codice");
        cJSON_Delete(root);
        exit(-1);
    }

    cJSON *version_obj = cJSON_GetObjectItemCaseSensitive(root, version_key);
    if (!version_obj) {
        version_obj = cJSON_CreateObject();
        cJSON_AddItemToObject(root, version_key, version_obj);
    }

    char vec_key[16];
    snprintf(vec_key, sizeof(vec_key), "%zu", vec);

    cJSON *vec_arr = cJSON_GetObjectItemCaseSensitive(version_obj, vec_key);
    if (!vec_arr) {
        vec_arr = cJSON_CreateArray();
        if (!vec_arr) {
            cJSON_Delete(root);
            return -1;
        }
        cJSON_AddItemToObject(version_obj, vec_key, vec_arr);
    }

    cJSON *entry = cJSON_CreateObject();
    if (!entry) {
        cJSON_Delete(root);
        return -1;
    }
    cJSON_AddNumberToObject(entry, "kernel_ms", round_to_3_dec(kernel_ms));
    cJSON_AddNumberToObject(entry, "h2d_ms", round_to_3_dec(h2d_ms));
    cJSON_AddNumberToObject(entry, "d2h_ms", round_to_3_dec(d2h_ms));
    cJSON_AddNumberToObject(entry, "cpu_post_ms", round_to_3_dec(cpu_post_ms));
    cJSON_AddNumberToObject(entry, "effective_bw", round_to_3_dec(effective_bw));
    cJSON_AddNumberToObject(entry, "useful_threads", useful_threads);

    cJSON_AddItemToArray(vec_arr, entry);

    char *rendered = cJSON_Print(root);
    cJSON_Delete(root);

    if (!rendered) return -2;

    fp = fopen(filepath, "wb");
    if (!fp) {
        free(rendered);
        return -3;
    }

    fputs(rendered, fp);
    fclose(fp);
    free(rendered);

    return 0;
}

#endif