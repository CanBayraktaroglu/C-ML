#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <stdlib.h>
#include <string.h>
#include "point.h"

typedef struct {
    double* data;
    size_t n_rows;
    size_t n_cols;
} Matrix;

void matrix_create(Matrix** mat ,size_t n_rows, size_t n_cols){
    *mat = (Matrix*)malloc(sizeof(Matrix));
    if (mat == NULL){
        printf("Failed to allocate memory for matrix.\n");
        exit(1);
    }
    (*mat)->data = (double*)calloc(n_rows * n_cols , sizeof(double));
    (*mat)->n_rows = n_rows;
    (*mat)->n_cols = n_cols;
};

void matrix_realloc(Matrix* mat, size_t n_rows, size_t n_cols){
    mat->data = (double*) realloc(mat->data, n_rows * n_cols * sizeof(double));
    mat->n_rows = n_rows;
    mat->n_cols = n_cols;
};

void matrix_destroy(Matrix* mat){
    if (mat == NULL) return;

    free(mat->data);
    mat->data = NULL;
};

void matrix_set(Matrix* mat, size_t i, size_t j, double val){
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

double matrix_get(Matrix* mat, size_t i, size_t j){
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
    double* data = (double*)malloc(mat->n_rows * mat->n_cols * sizeof(double));
    memcpy(data, mat->data, mat->n_rows * mat->n_cols * sizeof(double));
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
    Matrix* temp = NULL;
    matrix_create(&temp, mat->n_cols, mat->n_rows);
    for(size_t i = 0; i < mat->n_rows; i++){
        for(size_t j = 0; j < mat->n_cols; j++){
            matrix_set(temp, j, i, matrix_get(mat, i, j));
        }
    }
    return temp;
};

Matrix* scalar_product(Matrix* mat, double scalar, unsigned char free){
    Matrix* result = NULL;
    matrix_create(&result, mat->n_rows, mat->n_cols);
    for (size_t i = 0; i < mat->n_rows; i++){
        for (size_t j = 0; j < mat->n_cols; j++){
            matrix_set(result, i, j, scalar * matrix_get(mat, i, j));
        }
    }
    if (free) matrix_destroy(mat);
    return result;
};

void matrix_add(Matrix* a, Matrix* b, Matrix* output, unsigned char free){
    if (a == NULL){
        printf("Matrix a is pointing to an empty address\n.");
        return;
    }

    if (b == NULL){
        printf("Matrix b is pointing to an empty address\n.");
        return;
    }

    if (a->n_rows != b->n_rows || a->n_cols != b->n_cols){
        printf("Matrix dimensions do not match for addition\n.");
        exit(0);
    }

    // reallocate memory for output
    matrix_realloc(output, a->n_rows, a->n_cols);

    for (size_t i = 0; i < a->n_rows; i++){
        for (size_t j = 0; j < a->n_cols; j++){
            matrix_set(output, i, j, matrix_get(a, i, j) + matrix_get(b, i, j));
        }
    }
    if (free) {matrix_destroy(a); matrix_destroy(b);}
};

void matrix_subtract(Matrix* a, Matrix* b, Matrix* output, unsigned char free){
    if (a == NULL){
        printf("Matrix a is pointing to an empty address\n.");
        return;
    }

    if (b == NULL){
        printf("Matrix b is pointing to an empty address\n.");
        return;
    }

    if (a->n_rows != b->n_rows || a->n_cols != b->n_cols){
        printf("Matrix dimensions do not match for subtraction.\n");
        exit(0);
    }
    // reallocate memory for output
    matrix_realloc(output, a->n_rows, a->n_cols);
    
    for (size_t i = 0; i < a->n_rows; i++){
        for (size_t j = 0; j < a->n_cols; j++){
            matrix_set(output, i, j, matrix_get(a, i, j) - matrix_get(b, i, j));
        }
    }
    if (free) {matrix_destroy(a); matrix_destroy(b);}
};

void matrix_print(Matrix* mat){
    for(size_t i = 0; i < mat->n_rows; i++){
        for(size_t j = 0; j < mat->n_cols; j++){
            printf("%f ", matrix_get(mat, i, j));
        }
        printf("\n");
    }
};

void matrix_multiply(Matrix* a, Matrix* b, Matrix* output, unsigned char free){
    if (a == NULL){
        printf("Matrix a is pointing to an empty address\n.");
        return;
    }

    if (b == NULL){
        printf("Matrix b is pointing to an empty address\n.");
        return;
    }

    if (a->n_cols != b->n_rows){
        printf("Matrix dimensions do not match for multiplication.\n");
        exit(0);
    }
    
    // reallocate memory for output 
    matrix_realloc(output, a->n_rows, b->n_cols);    
    
    for (size_t i = 0; i < a->n_rows; i++){
        for (size_t j = 0; j < b->n_cols; j++){
            double sum = 0.0;
            for (size_t k = 0; k < a->n_cols; k++){
                sum += matrix_get(a, i, k) * matrix_get(b, k, j);
            }
            matrix_set(output, i, j, sum);
        }
    }

    if (free){ matrix_destroy(a); matrix_destroy(b);}
};


Matrix* create_identity_matrix(size_t n){
    Matrix* mat = NULL;
    matrix_create(&mat, n, n);
    for (size_t i = 0; i < n; i++){
        matrix_set(mat, i, i, 1);
    }
    return mat;
};
Matrix* matrix_copy(Matrix* mat){
    Matrix* copy = NULL;
    matrix_create(&copy, mat->n_rows, mat->n_cols);
    memcpy(copy->data, mat->data, mat->n_rows * mat->n_cols * sizeof(double));
    return copy;
};

double matrix_froebenius_norm(Matrix* mat){
    double sum = 0.0;
    for (size_t i = 0; i < mat->n_rows; i++){
        for (size_t j = 0; j < mat->n_cols; j++){
            sum += matrix_get(mat, i, j) * matrix_get(mat, i, j);
        }
    }
    return sqrt(sum);
};

/* double calculate_inverse_tolerance(Matrix* A_T, Matrix* x){
    Matrix* prod = matrix_multiply(A_T, x, 0);
    matrix_print(x);
    Matrix* identity = create_identity_matrix(A_T->n_rows);
    Matrix* diff = matrix_subtract(prod, identity, 1);
    double tol = matrix_froebenius_norm(diff);
    matrix_destroy(&diff);
    return tol;
} */

/* Matrix* matrix_inverse_newton(Matrix* mat, size_t max_iter, double tol, unsigned char free){
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
 */

Matrix* matrix_create_from_array(size_t n_rows, size_t n_cols, double (*arr)[n_cols]){
    if (arr == NULL){
        printf("Array to be converted to matrix points to an empty address.\n");
        exit(0);
    }
    Matrix* mat = NULL;
    matrix_create(&mat, n_rows, n_cols);
    for (size_t i = 0; i < n_rows; i++){
        for (size_t j = 0; j < n_cols; j++){
            matrix_set(mat, i, j, arr[i][j]);
        }
    }

    return mat;
}

#endif // __MATRIX_H__