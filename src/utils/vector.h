#ifndef VECTOR_H_
#define VECTOR_H_

#include "point.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>
#include <string.h>
#include <time.h>

typedef struct{
	Point* data;
	unsigned short int size;
	unsigned short int capacity;
}Vector;

Vector* vector_create(unsigned short int initial_capacity) {
    Vector* vec = (Vector*) malloc(sizeof(Vector));
    vec->data = (Point*) malloc(initial_capacity * sizeof(Point));
    vec->size = 0;
    vec->capacity = initial_capacity;
    return vec;
};

Point* vector_at(Vector* vec, unsigned short int index) {
    if (index < 0 || index >= vec->size) {
        printf("Index out of bounds for idx: %hu\n", index);
        printf("Size: %hu\n", vec->size);
        exit(1);
    }
    return &vec->data[index];
};
void vector_destroy(Vector** vec) {
    printf("Destroying vector\n");
    if (*vec == NULL) return;

    free((*vec)->data);
    (*vec)->data = NULL;
    free(*vec);
    *vec = NULL;
    
};

void vector_destroy_all(Vector* vec) {
    //printf("Destroying vector all\n");
    if (vec == NULL) return;
    
    for (unsigned short int i = 0; i < vec->size; i++){
        Point* p = vec->data + i;
    	color_destroy(p->color);
    }
    free(vec->data);
    vec->data = NULL;
    free(vec);
    vec = NULL;
    
};

void vector_push_back(Vector* vec, Point* element) {
    if (vec->size == vec->capacity) {
        vec->capacity<<=1;
        vec->data = (Point*) realloc(vec->data, (unsigned long)vec->capacity * sizeof(Point));
    }
    memcpy(vec->data + vec->size++, element, sizeof(Point));
    
};


Point* vector_pop(Vector* vec){
	Point* p = vector_at(vec, vec->size - 1);
	free(&vec->data[vec->size - 1]);
	vec->size--;
	return p;

};

void vector_print(Vector* vec) {
    printf("[");
    for (unsigned short int i = 0; i < vec->size; i++) {
	printf("[");

	Point* p_pt = vec->data + i;	
    for (unsigned char j = 0; j < p_pt->dim; j++){
        printf("%f", p_pt->point[j]);
        if (j < p_pt->dim - 1) {
            printf(", ");
        }
    }
	printf("]");

        if (i < vec->size - 1) {
            printf(", ");
        }
    }
    printf("]\n");
};

void vector_shuffle(Vector* vec) {
    for (unsigned short int i = 0; i < vec->size; i++) {
        unsigned short int j = rand() % vec->size;
        Point temp = vec->data[i];
        vec->data[i] = vec->data[j];
        vec->data[j] = temp;
    }
};  


#endif