
#ifndef UTILS_H_
#define UTILS_H_

#include "vector.h"
#include "math.h"
#include "time.h"
#include "stdio.h"
#include "stdlib.h"
#include "yaml.h"
#include "matrix.h"

#define TABLE_SIZE 10

typedef struct {
    float split_ratio;
    unsigned char k;
    char data_path[256];
} KNN_Config;

typedef struct{
    char data_path[256];
    float split_ratio;
}DT_Config;

typedef struct {
    unsigned short int num_classes;
    float accuracy;
    float* precision;
} Metrics;

typedef struct{
    float dist;
    struct KDTreeNode* node;
}HeapNode;

typedef struct{
    HeapNode* nodes;
    unsigned short int size;
    unsigned short int capacity;
}MaxHeap;

typedef struct HashEntry {
    void* key;
    void* value;
    struct HashEntry* next;
} HashEntry;

typedef struct HashTable {
    HashEntry** entries;
    int (*hash_func)(const void* key);
    int (*key_cmp)(const void* key1, const void* key2);
} HashTable;

void print_metrics(Metrics* metrics){
    printf("Accuracy: %f\n", metrics->accuracy); 
    for (unsigned short int i = 0; i < metrics->num_classes; i++){
        printf("Class %hu\n", i);    
        printf("Precision: %f\n", *(metrics->precision + i));
    }
};

Metrics* create_metrics(unsigned short int num_classes){
    Metrics* metrics = (Metrics*)malloc(sizeof(Metrics));
    metrics->num_classes = num_classes;
    metrics->accuracy = 0.0;
    metrics->precision = (float*)malloc(sizeof(float) * num_classes);
    return metrics;
};

void metrics_destroy(Metrics** metrics){
    free((*metrics)->precision);
    (*metrics)->precision = NULL;
    free(*metrics);
    *metrics = NULL;
};

HashEntry* create_entry(void *key, void *value) {
    HashEntry *entry = (HashEntry *)malloc(sizeof(HashEntry));
    entry->key = key;
    entry->value = value;
    entry->next = NULL;
    return entry;
};

HashTable* create_table(int (*hash_func)(const void *key), int (*key_cmp)(const void *key1, const void *key2)) {
    HashTable *table = (HashTable *)malloc(sizeof(HashTable));
    table->entries = (HashEntry **)malloc(sizeof(HashEntry *) * TABLE_SIZE);
    for (int i = 0; i < TABLE_SIZE; ++i) {
        table->entries[i] = NULL;
    }
    table->hash_func = hash_func;
    table->key_cmp = key_cmp;
    return table;
};

void ht_insert(HashTable *table, void *key, void *value) {
    int slot = table->hash_func(key);

    HashEntry* entry = table->entries[slot];

    // No entry means slot is empty, insert immediately
    if (entry == NULL) {
        table->entries[slot] = create_entry(key, value);
        return;
    }

    HashEntry *prev;

    // Walk through the chain (collision resolution by chaining)
    while (entry != NULL) {
        // If key already exists, update value
        if (!table->key_cmp(entry->key, key)) {
            entry->value = value;  // Assuming ownership is managed elsewhere
            return;
        }

        prev = entry;
        entry = entry->next;
    }

    // If key not found, add new entry at the end of the chain
    prev->next = create_entry(key, value);
};

void* ht_search(HashTable *table, void *key) {
    int slot = table->hash_func(key);

    HashEntry *entry = table->entries[slot];

    // Walk through the chain
    while (entry != NULL) {
        if (!table->key_cmp(entry->key, key)) {
            return entry->value;
        }
        entry = entry->next;
    }

    printf("Key not found\n");
    return NULL;
};

void ht_increment(HashTable *table, void *key) {
    
    int slot = table->hash_func(key);

    HashEntry *entry = table->entries[slot];
    
    // Walk through the chain
    while (entry != NULL) {
        if (!table->key_cmp(entry->key, key)) {
            (*(int*)entry->value)++;  // Assuming value is an integer (int*
            return;
        }
        entry = entry->next;
    }

    // If key not found, insert it with an initial value of 1
    int *initial_value = (int *)malloc(sizeof(int));
    *initial_value = 1;
    ht_insert(table, key, initial_value);

};

void ht_delete(HashTable *table, void *key) {
    unsigned int slot = table->hash_func(key);

    HashEntry *entry = table->entries[slot];
    HashEntry *prev = NULL;

    // Walk through the chain
    while (entry != NULL) {
        if (table->key_cmp(entry->key, key) == 0) {
            if (prev == NULL) {
                table->entries[slot] = entry->next;
            } else {
                prev->next = entry->next;
            }
            free(entry);
            return;
        }
        prev = entry;
        entry = entry->next;
    }
};

void ht_destroy(HashTable** table) {
    if (*table == NULL) return;
    for (int i = 0; i < TABLE_SIZE; ++i) {
        HashEntry *entry = (*table)->entries[i];
        while (entry != NULL) {
            HashEntry *temp = entry;
            free(temp->value);
            entry = entry->next;
            free(temp);
        }
    }
    free((*table)->entries);
    free(*table);
};

int int_hash(const void *key) {
    return (*(int *)key) % TABLE_SIZE;
;}

int int_cmp(const void *key1, const void *key2) {
    return (*(int *)key1) - (*(int *)key2);
};

unsigned int str_hash(const char* key) {
    unsigned long int value = 0;
    unsigned int i = 0;
    unsigned int key_len = strlen(key);

    // Convert the string to an integer
    for (; i < key_len; ++i) {
        value = value * 37 + key[i];
    }

    // Ensure the value is within the range of the table size
    return value % TABLE_SIZE;
};

 // return a uniformly distributed random number
float randomGenerator()
{
  return rand() / (float)RAND_MAX;
};

 // return a normally distributed random number
float normalRandom()
{
  float y1 = randomGenerator();
  float y2 = randomGenerator();
  return cos(2*3.14*y2) * sqrt(-2.*log(y1));
};

void swap_nodes(HeapNode* a, HeapNode* b){
    HeapNode tmp = *a;
    *a = *b;
    *b = tmp;
};

void max_heapify(MaxHeap* heap, unsigned short int idx){
    unsigned short int largest = idx;
    unsigned short int left = 2 * idx + 1;
    unsigned short int right = 2 * idx + 2;

    if (left < heap->size && heap->nodes[left].dist > heap->nodes[largest].dist){
        largest = left;
    }

    if (right < heap->size && heap->nodes[right].dist > heap->nodes[largest].dist){
        largest = right;
    }

    if (largest != idx){
        swap_nodes(&heap->nodes[idx], &heap->nodes[largest]);
        max_heapify(heap, largest);
    }
};

void insert_max_heap(MaxHeap* heap, HeapNode* hn){
    if (heap->size < heap->capacity) {
        *(heap->nodes + heap->size) = *hn;
        int i = (int)heap->size++;
        while (i != 0 && heap->nodes[(i - 1) / 2].dist < heap->nodes[i].dist) {
            swap_nodes(&heap->nodes[i], &heap->nodes[(i - 1) / 2]);
            i = (i - 1) / 2;
        }
    } else if (hn->dist < heap->nodes[0].dist) {
        *heap->nodes = *hn;
        max_heapify(heap, 0);
    }
};

Point* choose_pivot(Vector* v){
    Point* p = v->data + (unsigned short)(v->size / 2);
    return p;
};

void swap(Vector* v, int i, int j){
    Point tmp = *vector_at(v, i);
    *vector_at(v, i) = *vector_at(v, j);
    *vector_at(v, j) = tmp;
};

int partition_(Vector* v, int low, int high, int dim){
    Point* pivot = vector_at(v, high);//choose_pivot(v);
    int i = low;
    //printf("low, high: %i,%i,%i\n", low, high, dim);

    for (int j = low; j < high; j++){
        if (vector_at(v, j)->point[dim] <= pivot->point[dim]){ 
            //printf("Swapping %i and %i\n", i, j);
            swap(v, i, j);
            i++;
        }
    }
    swap(v, i, high);
    return i;
};

void qsort_(Vector* v, int low, int high, int dim){
    if (low < high){
        int pivot_idx = partition_(v, low, high, dim);
        qsort_(v, low, pivot_idx - 1, dim);
        qsort_(v, pivot_idx + 1, high, dim);
    }
};

void load_yaml_knn(const char *filepath, KNN_Config *config) {
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "Could not open file: %s\n", filepath);
        return;
    }

    yaml_parser_t parser;
    yaml_event_t event;
    int done = 0;

    if (!yaml_parser_initialize(&parser)) {
        fputs("Failed to initialize parser!\n", stderr);
        fclose(file);
        return;
    }

    yaml_parser_set_input_file(&parser, file);

    char *current_key = NULL;

    while (!done) {
        if (!yaml_parser_parse(&parser, &event)) {
            fprintf(stderr, "Parser error %d\n", parser.error);
            break;
        }

        switch (event.type) {
            case YAML_MAPPING_START_EVENT:
                break;
            case YAML_MAPPING_END_EVENT:
                break;
            case YAML_SCALAR_EVENT:
                if (current_key == NULL) {
                    current_key = strdup((char *)event.data.scalar.value);
                } else {
                    if (strcmp(current_key, "split_ratio") == 0) {
                        config->split_ratio = atof((char *)event.data.scalar.value);
                    } else if (strcmp(current_key, "k") == 0) {
                        config->k = atoi((char *)event.data.scalar.value);
                    } else if (strcmp(current_key, "data_path") == 0) {
                        strncpy(config->data_path, (char *)event.data.scalar.value, sizeof(config->data_path) - 1);
                        config->data_path[sizeof(config->data_path) - 1] = '\0'; // Ensure null-termination
                    }
                    free(current_key);
                    current_key = NULL;
                }
                break;
            case YAML_STREAM_END_EVENT:
                done = 1;
                break;
            default:
                break;
        }

        yaml_event_delete(&event);
    }

    if (current_key) {
        free(current_key);
    }

    yaml_parser_delete(&parser);
    fclose(file);
};

void load_yaml_dt(const char *filepath, DT_Config *config) {
    FILE *file = fopen(filepath, "rb");
    if (!file) {
        fprintf(stderr, "Could not open file: %s\n", filepath);
        return;
    }

    yaml_parser_t parser;
    yaml_event_t event;
    int done = 0;

    if (!yaml_parser_initialize(&parser)) {
        fputs("Failed to initialize parser!\n", stderr);
        fclose(file);
        return;
    }

    yaml_parser_set_input_file(&parser, file);

    char *current_key = NULL;

    while (!done) {
        if (!yaml_parser_parse(&parser, &event)) {
            fprintf(stderr, "Parser error %d\n", parser.error);
            break;
        }

        switch (event.type) {
            case YAML_MAPPING_START_EVENT:
                break;
            case YAML_MAPPING_END_EVENT:
                break;
            case YAML_SCALAR_EVENT:
                if (current_key == NULL) {
                    current_key = strdup((char *)event.data.scalar.value);
                } else {
                    if (strcmp(current_key, "split_ratio") == 0) 
                        config->split_ratio = atof((char *)event.data.scalar.value);
                   else if (strcmp(current_key, "data_path") == 0) {
                        strncpy(config->data_path, (char *)event.data.scalar.value, sizeof(config->data_path) - 1);
                        config->data_path[sizeof(config->data_path) - 1] = '\0'; // Ensure null-termination
                    }
                    free(current_key);
                    current_key = NULL;
                }
                break;
            case YAML_STREAM_END_EVENT:
                done = 1;
                break;
            default:
                break;
        }

        yaml_event_delete(&event);
    }

    if (current_key) {
        free(current_key);
    }

    yaml_parser_delete(&parser);
    fclose(file);
};

size_t* calculate_class_frequency(Vector* vec, unsigned short num_classes){
    size_t* class_freq = (size_t*)malloc(sizeof(size_t) * (size_t)num_classes);

    // Initialize the class frequency array
    for (unsigned short i = 0; i < num_classes; i++){
        class_freq[i] = 0;
    }

    for (unsigned short i = 0; i < vec->size; i++){
        Point* p = vector_at(vec, i);
        class_freq[p->class]++;
    }
    return class_freq;
};

unsigned short calculate_num_classes(Vector* vec, unsigned short total_num_classes){
    size_t* class_freq = calculate_class_frequency(vec, total_num_classes);
    unsigned short num_classes = 0;

    for(unsigned short i = 0; i < total_num_classes; i++){
        if (class_freq[i] > 0){
            num_classes++;
        }
    }

    free(class_freq);
    class_freq = NULL;
    return num_classes;
};

float calculate_entropy(Vector* vec, unsigned char num_classes){	
    size_t* class_freq = calculate_class_frequency(vec, num_classes);
    float entropy = 0.0f;
    float total = (float)vec->size;

    if (total == 0.0){
        free(class_freq);
        class_freq = NULL;
        return 1.0;
    } 

    for (unsigned short i = 0; i < num_classes; i++){
        if (!class_freq[i]) continue;
        entropy += -((float)class_freq[i] / total) * log2f((float)class_freq[i] / total);
    }

    free(class_freq);
    class_freq = NULL;
    return entropy;
};

float calculate_info_gain(Vector* parent, Vector* left, Vector* right, unsigned char num_classes){
    float parent_entropy = 0;
    float left_entropy = 0;
    float right_entropy = 0;
    float total = (float)parent->size;

    parent_entropy = calculate_entropy(parent, num_classes);
    left_entropy = calculate_entropy(left, num_classes);
    right_entropy = calculate_entropy(right, num_classes);

    float info_gain = parent_entropy - ((float)left->size / total) * left_entropy - ((float)right->size / total) * right_entropy;
    return info_gain;
};

float* dot_product(float* a, float* b, size_t n){
    float* result = (float*)calloc(n, sizeof(float));
    for (size_t i = 0; i < n; i++){
        result[i] = a[i] * b[i];
    }
    return result;
};
Matrix* outer_product(float* a, float* b, size_t n){
    Matrix* matrix = matrix_create(n, n);
    for (size_t i = 0; i < n; i++){
        for (size_t j = 0; j < n; j++){
            matrix_set(matrix, i, j, a[i] * b[j]);
        }
    }
    return matrix;
};

Matrix* scalar_product(Matrix* mat, float scalar){
    Matrix* result = matrix_create(mat->n_rows, mat->n_cols);
    for (size_t i = 0; i < mat->n_rows; i++){
        for (size_t j = 0; j < mat->n_cols; j++){
            matrix_set(result, i, j, scalar * matrix_get(mat, i, j));
        }
    }
    return result;
};

Matrix* matrix_add(Matrix* a, Matrix* b){
    if (a->n_rows != b->n_rows || a->n_cols != b->n_cols){
        printf("Matrix dimensions do not match\n");
        return NULL;
    }
    Matrix* result = matrix_create(a->n_rows, a->n_cols);
    for (size_t i = 0; i < a->n_rows; i++){
        for (size_t j = 0; j < a->n_cols; j++){
            matrix_set(result, i, j, matrix_get(a, i, j) + matrix_get(b, i, j));
        }
    }
    return result;
};

Matrix* matrix_subtract(Matrix* a, Matrix* b){
    if (a->n_rows != b->n_rows || a->n_cols != b->n_cols){
        printf("Matrix dimensions do not match\n");
        return NULL;
    }
    Matrix* result = matrix_create(a->n_rows, a->n_cols);
    for (size_t i = 0; i < a->n_rows; i++){
        for (size_t j = 0; j < a->n_cols; j++){
            matrix_set(result, i, j, matrix_get(a, i, j) - matrix_get(b, i, j));
        }
    }
    return result;
};

Matrix* matrix_multiply(Matrix* a, Matrix* b){
    if (a->n_cols != b->n_rows){
        printf("Matrix dimensions do not match\n");
        return NULL;
    }
    
    Matrix* result = matrix_create(a->n_rows, b->n_cols);
    for (size_t i = 0; i < a->n_rows; i++){
        for (size_t j = 0; j < b->n_cols; j++){
            float sum = 0.0;
            for (size_t k = 0; k < a->n_cols; k++){
                sum += matrix_get(a, i, k) * matrix_get(b, k, j);
            }
            matrix_set(result, i, j, sum);
        }
    }
    return result;
};

#endif