#ifndef MY_UTILITIES
#define MY_UTILITIES

#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <dirent.h>
#include <sys/stat.h>
#include <ctype.h>
#include <time.h>
#include <math.h>
#include "../cJSON.h"

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

typedef struct {
    float *data;
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
    r->data = (float *)malloc(capacity * sizeof(float));
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
                float *temp = (float *)realloc(r->data, capacity * sizeof(float));
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

            r->data[r->total_cells++] = val;
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

    float *shrink = (float *)realloc(r->data, r->total_cells * sizeof(float));
    if (shrink) {
        r->data = shrink;
    }

    return r;
}

static inline void free_raster_list(Raster **raster_list, int count) {
    if (raster_list == NULL) return;

    for (int i = 0; i < count; ++i) {
        if (raster_list[i] != NULL) {
            free(raster_list[i]->data);
            free(raster_list[i]);
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

int save_overlap_matrix_to_file(const char *filename, const float *matrix, size_t rows_flow, size_t cols_habi) {
    if (matrix == NULL || filename == NULL) {
        return 0;
    }

    FILE *fp = fopen(filename, "w");
    if (fp == NULL) {
        perror("Errore apertura file di output");
        return 0;
    }

    fprintf(fp, "Flow\\Hab");
    for (size_t j = 0; j < cols_habi; ++j) {
        fprintf(fp, "\tH_%zu", j);
    }
    fprintf(fp, "\n");

    for (size_t i = 0; i < rows_flow; ++i) {
        fprintf(fp, "F_%zu", i);
        for (size_t j = 0; j < cols_habi; ++j) {
            fprintf(fp, "\t%.6f", matrix[i * cols_habi + j]);
        }
        fprintf(fp, "\n");
    }

    fclose(fp);
    return 1;
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
    double kernel_ms,
    double useful_threads
) {
    size_t vec = 1;

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
        fprintf(stderr, "nome lista json sbagliato, lato codice\n");
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

    // Recupera l'array se esiste già, altrimenti lo crea
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