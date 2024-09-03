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
    if (i >= mat->n_rows){
        printf("row index exceeded matrix row number\n.");
        exit(0);
    }

    if (j >= mat->n_cols){
        printf("col index exceeded matrix col number\n.");
        exit(0);
    }
    mat->data[i * mat->n_cols + j] = val;
};

float matrix_get(Matrix* mat, size_t i, size_t j){
    if (i >= mat->n_rows){
        printf("row index exceeded matrix row number\n.");
        exit(0);
    }

    if (j >= mat->n_cols){
        printf("col index exceeded matrix col number\n.");
        exit(0);
    }

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

Matrix* scalar_product(Matrix* mat, float scalar, unsigned char free){
    Matrix* result = matrix_create(mat->n_rows, mat->n_cols);
    for (size_t i = 0; i < mat->n_rows; i++){
        for (size_t j = 0; j < mat->n_cols; j++){
            matrix_set(result, i, j, scalar * matrix_get(mat, i, j));
        }
    }
    if (free) matrix_destroy(&mat);
    return result;
};

Matrix* matrix_add(Matrix* a, Matrix* b, unsigned char free){
    if (a->n_rows != b->n_rows || a->n_cols != b->n_cols){
        printf("Matrix dimensions do not match for addition\n.");
        exit(0);
    }
    Matrix* result = matrix_create(a->n_rows, a->n_cols);
    for (size_t i = 0; i < a->n_rows; i++){
        for (size_t j = 0; j < a->n_cols; j++){
            matrix_set(result, i, j, matrix_get(a, i, j) + matrix_get(b, i, j));
        }
    }
    if (free) matrix_destroy(&a); matrix_destroy(&b);
    return result;
};

Matrix* matrix_subtract(Matrix* a, Matrix* b, unsigned char free){
    if (a->n_rows != b->n_rows || a->n_cols != b->n_cols){
        printf("Matrix dimensions do not match for subtraction.\n");
        exit(0);
    }
    Matrix* result = matrix_create(a->n_rows, a->n_cols);
    for (size_t i = 0; i < a->n_rows; i++){
        for (size_t j = 0; j < a->n_cols; j++){
            matrix_set(result, i, j, matrix_get(a, i, j) - matrix_get(b, i, j));
        }
    }
    if (free) {matrix_destroy(&a); matrix_destroy(&b);}
    return result;
};

void matrix_print(Matrix* mat){
    for(size_t i = 0; i < mat->n_rows; i++){
        for(size_t j = 0; j < mat->n_cols; j++){
            printf("%f ", matrix_get(mat, i, j));
        }
        printf("\n");
    }
};

Matrix* matrix_multiply(Matrix* a, Matrix* b, unsigned char free){
    if (a->n_cols != b->n_rows){
        printf("Matrix dimensions do not match for multiplication.\n");
        exit(0);
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

    if (free){ matrix_destroy(&a); matrix_destroy(&b);}

    return result;
};


Matrix* create_identity_matrix(size_t n){
    Matrix* mat = matrix_create(n, n);
    for (size_t i = 0; i < n; i++){
        matrix_set(mat, i, i, 1);
    }
    return mat;
};
Matrix* matrix_copy(Matrix* mat){
    Matrix* copy = matrix_create(mat->n_rows, mat->n_cols);
    memcpy(copy->data, mat->data, mat->n_rows * mat->n_cols * sizeof(float));
    return copy;
};

float matrix_froebenius_norm(Matrix* mat){
    float sum = 0.0;
    for (size_t i = 0; i < mat->n_rows; i++){
        for (size_t j = 0; j < mat->n_cols; j++){
            sum += matrix_get(mat, i, j) * matrix_get(mat, i, j);
        }
    }
    return sqrt(sum);
};

float calculate_inverse_tolerance(Matrix* A_T, Matrix* x){
    Matrix* prod = matrix_multiply(A_T, x, 0);
    matrix_print(x);
    Matrix* identity = create_identity_matrix(A_T->n_rows);
    Matrix* diff = matrix_subtract(prod, identity, 1);
    float tol = matrix_froebenius_norm(diff);
    matrix_destroy(&diff);
    return tol;
}

Matrix* matrix_inverse_newton(Matrix* mat, size_t max_iter, float tol, unsigned char free){
    Matrix* X = matrix_create(mat->n_rows, mat->n_cols);

    // Initialize x_prev with 1.0s
    printf("Initializing matrix with 1.0s.\n");
    for (size_t i = 0; i < X->n_rows; i++){
        for (size_t j = 0; j < X->n_cols; j++){
            matrix_set(X, i, j, 1.0f);
        }
    }
 
    size_t iter = 0;
    Matrix* X_T = matrix_transpose(mat);
    Matrix* scaled_identity = scalar_product(create_identity_matrix(mat->n_cols), 2.0f, 1);
    printf("INITIAL X_T_n_cols: %lu, X_n_rows: %lu\n", X_T->n_cols, X->n_rows);
    while (iter <= max_iter && calculate_inverse_tolerance(X_T, X) > tol){
        printf("Loop %lu.\n", iter);
        printf("X_T_n_cols: %lu, X_n_rows: %lu\n", X_T->n_cols, X->n_rows);

        X = matrix_multiply(X, matrix_subtract(scaled_identity, matrix_multiply(X_T, X, 0), 0), 0);
        iter++;
    }

    matrix_destroy(&scaled_identity);
    matrix_destroy(&X_T);
    if (free) matrix_destroy(&mat);

    return X;
};

#endif // __MATRIX_H__