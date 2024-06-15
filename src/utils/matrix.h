#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <stdlib.h>
#include <string.h>
#include "point.h"

typedef struct {
    float* data;
    size_t n_rows;
    size_t n_cols;
} Matrix;

Matrix* matrix_create(size_t n_rows, size_t n_cols){
    Matrix* mat = (Matrix*)malloc(sizeof(Matrix));
    mat->data = (float*)calloc(n_rows * n_cols , sizeof(float));
    mat->n_rows = n_rows;
    mat->n_cols = n_cols;
    return mat;
};

void matrix_destroy(Matrix** mat){
    free((*mat)->data);
    (*mat)->data = NULL;
    free(*mat);
    *mat = NULL;
};

void matrix_set(Matrix* mat, size_t i, size_t j, float val){
    mat->data[i * mat->n_cols + j] = val;
};

float matrix_get(Matrix* mat, size_t i, size_t j){
    return mat->data[i * mat->n_cols + j];
};

void matrix_transpose_inplace(Matrix* mat){
    float* data = (float*)malloc(mat->n_rows * mat->n_cols * sizeof(float));
    memcpy(data, mat->data, mat->n_rows * mat->n_cols * sizeof(float));
    for(size_t i = 0; i < mat->n_rows; i++){
        for(size_t j = 0; j < mat->n_cols; j++){
            mat->data[j * mat->n_rows + i] = data[i * mat->n_cols + j];
        }
    }
    size_t temp = 0;
    temp = mat->n_rows;
    mat->n_rows = mat->n_cols;
    mat->n_cols = temp;
    free(data);
};

Matrix* matrix_transpose(Matrix* mat){
    Matrix* temp = matrix_create(mat->n_cols, mat->n_rows);
    for(size_t i = 0; i < mat->n_rows; i++){
        for(size_t j = 0; j < mat->n_cols; j++){
            matrix_set(temp, j, i, matrix_get(mat, i, j));
        }
    }
    return temp;
};

void matrix_print(Matrix* mat){
    for(size_t i = 0; i < mat->n_rows; i++){
        for(size_t j = 0; j < mat->n_cols; j++){
            printf("%f ", matrix_get(mat, i, j));
        }
        printf("\n");
    }
};
#endif // __MATRIX_H__