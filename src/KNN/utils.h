
#ifndef UTILS_H_
#define UTILS_H_

#include "vector.h"
#include "k_d_tree.h"
#include "math.h"
#include "time.h"
#include "stdio.h"
#include "stdlib.h"

#define TABLE_SIZE 10

typedef struct {
    unsigned short int num_classes;
    float* accuracy;
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
    for (unsigned short int i = 0; i < metrics->num_classes; i++){
        printf("Class %hu\n", i);
        printf("Accuracy: %f\n", *(metrics->accuracy + i));
        printf("Precision: %f\n", *(metrics->precision + i));
    }
};

void metrics_destroy(Metrics* metrics){
    free(metrics->accuracy);
    free(metrics->precision);
    metrics->accuracy = NULL;
    metrics->precision = NULL;
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

void ht_destroy(HashTable *table) {
    for (int i = 0; i < TABLE_SIZE; ++i) {
        HashEntry *entry = table->entries[i];
        while (entry != NULL) {
            HashEntry *temp = entry;
            free(temp->value);
            entry = entry->next;
            free(temp);
        }
    }
    free(table->entries);
    free(table);
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

signed char compareX(const void* a, const void* b){
    Point* pt1 = (Point*) a;
    Point* pt2 =  (Point*) b;

    if (*(pt1->point) < *(pt2->point)) return -1;
    if (*(pt1->point) > *(pt2->point)) return 1;

    return 0;

};

signed char compareY(const void* a, const void* b){
    Point* pt1 = (Point*) a;
    Point* pt2 = (Point*) b;

    float y1 = *(pt1->point+1);
    float y2 = *(pt2->point+1);

    if (y1 < y2) return -1;
    if (y1 > y2) return 1;

    return 0;


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

/* int partition_x(Vector* v, int low, int high){
    Point* pivot = vector_at(v, high);
    int i = low - 1;

    for (int j = low; j <= high; j++){
        if (vector_at(v, j)->point[0] < pivot->point[0]){ 
            i++;
            swap(v, i, j);
        }
    }
    swap(v, i+1, high);
    return i + 1;

}; */

/* int partition_y(Vector* v, int low, int high){
    Point* pivot = vector_at(v, high);
    int i = low - 1;

    for (int j = low; j <= high; j++){
        if (vector_at(v, j)->point[1] < pivot->point[1]){ 
            i++;
            swap(v, i, j);
        }
    }
    swap(v, i+1, high);
    return i + 1;

}; */

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


/* void qsort_x(Vector* v, int low, int high){
    if (low < high){
        int pivot_idx = partition_x(v, low, high);
        qsort_x(v, low, pivot_idx - 1);
        qsort_x(v, pivot_idx + 1, high);
    }
};

void qsort_y(Vector* v, int low, int high){
    if (low < high){
        int pivot_idx = partition_y(v, low, high);
        qsort_y(v, low, pivot_idx - 1);
        qsort_y(v, pivot_idx + 1, high);
    }
}; */

#endif