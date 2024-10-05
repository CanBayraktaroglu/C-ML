#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdlib.h>
#include <string.h>
#include "point.h"
#include "autodifferentation.h"

typedef struct Tensor{
    struct Tensor* self;
    ADNode** nodes;
    size_t n_rows;
    size_t n_cols;
    void (*realloc)(Tensor* self, const size_t n_rows, const size_t n_cols);
    void (*destroy)(Tensor* self);
    static double (*get_val)(Tensor* self, const size_t i, const size_t j);
    static double (*get_grad)(Tensor* self, const size_t i, const size_t j);
    static void (*set_val)(Tensor* self, const size_t i, const size_t j, const double val);
    static void (*set_grad)(Tensor* self, const size_t i, const size_t j, const double grad);
} Tensor;


void tensor_realloc(Tensor* self, const size_t n_rows, const size_t n_cols){
    self->nodes = (ADNode**) realloc(self->nodes, n_rows * n_cols * sizeof(ADNode*));
    self->n_rows = n_rows;
    self->n_cols = n_cols;
};

void tensor_destroy(Tensor* self){
    if (self == NULL) return;

    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            const size_t idx = i * self->n_cols + j;
            ADNode* node = self->nodes[idx];
            node->free(node);

        }
    }
    free(self->nodes);
    free(self);
};

void tensor_set_val(Tensor* self, const size_t i, const size_t j, const double val){
    if (i >= self->n_rows){
        printf("row index exceeded tensor row number.\n");
        exit(0);
    }

    if (j >= self->n_cols){
        printf("col index exceeded tensor col number.\n");
        exit(0);
    }
    ADNode* node = self->nodes[i * self->n_cols + j];
    node->set_val(node, val);

};

void tensor_set_grad(Tensor* self, const size_t i, const size_t j, const double grad){
    if (i >= self->n_rows){
        printf("row index exceeded tensor row number.\n");
        exit(0);
    }
    
    if (j >= self->n_cols){
        printf("col index exceeded tensor col number.\n");
        exit(0);
    }
    ADNode* node = self->nodes[i * self->n_cols + j];
    node->set_grad(node, grad);
};

static double tensor_get_val(Tensor* self, const size_t i, const size_t j){
    if (i >= self->n_rows){
        printf("row index exceeded tensor row number.\n");
        exit(0);
    }
    
    if (j >= self->n_cols){
        printf("col index exceeded tensor col number.\n");
        exit(0);
    }

    ADNode* node = self->nodes[i * self->n_cols + j];
    return node->get_val(node);

};

static double tensor_get_grad(Tensor* self, const size_t i, const size_t j){
    if (i >= self->n_rows){
        printf("row index exceeded tensor row number.\n");
        exit(0);
    }
    
    if (j >= self->n_cols){
        printf("col index exceeded tensor col number.\n");
        exit(0);
    }

    ADNode* node = self->nodes[i * self->n_cols + j];
    return node->get_grad(node);

};
void tensor_transpose_inplace(Tensor* self){
    ADNode* temp = (ADNode*)malloc(sizeof(ADNode));
    
    for(size_t i = 0; i < self->n_rows; i++){
        for(size_t j = 0; j < self->n_cols; j++){
            ADNode* source = self->nodes[i * self->n_cols + j]; // iter cols
            ADNode* target = self->nodes[j * self->n_rows + i];

            memcpy(temp, source, sizeof(ADNode));
            memcpy(source, target, sizeof(ADNode));
            memcpy(target, temp, sizeof(ADNode));
        }
    }
    
    temp->free(temp);        
};

Tensor* tensor_transpose(Tensor* self){
    Tensor* temp = tensor_new(self->n_cols, self->n_rows);

    for(size_t i = 0; i < self->n_rows; i++){
        for(size_t j = 0; j < self->n_cols; j++){
            ADNode* source = self->nodes[i * self->n_cols + j]; // iter cols
            ADNode* target = temp->nodes[j * self->n_rows + i];
            memcpy(target, source, sizeof(ADNode));
        }
    }

    return temp;
};

void tensor_scalar_product(Tensor* self, const double scalar){
    

    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            Tensor_set(*output, i, j, scalar * Tensor_get(mat, i, j));
        }
    }

};

void Tensor_add(Tensor* a, Tensor* b, Tensor** output, const unsigned char free){
    if (a == NULL){
        printf("Tensor a is pointing to an empty address\n.");
        return;
    }

    if (b == NULL){
        printf("Tensor b is pointing to an empty address\n.");
        return;
    }

    if (a->n_rows != b->n_rows || a->n_cols != b->n_cols){
        printf("Tensor dimensions do not match for addition\n.");
        exit(0);
    }
    if (*output == NULL){
        Tensor_create(output, a->n_rows, a->n_cols);
    }
    else {
        // reallocate memory for output
        Tensor_realloc(*output, a->n_rows, a->n_cols);
    }

    for (size_t i = 0; i < a->n_rows; i++){
        for (size_t j = 0; j < a->n_cols; j++){
            Tensor_set(*output, i, j, Tensor_get(a, i, j) + Tensor_get(b, i, j));
        }
    }
    if (free) {Tensor_destroy(a); Tensor_destroy(b);}
};

void Tensor_subtract(Tensor* a, Tensor* b, Tensor** output, const unsigned char free){
    if (a == NULL){
        printf("Tensor a is pointing to an empty address\n.");
        return;
    }

    if (b == NULL){
        printf("Tensor b is pointing to an empty address\n.");
        return;
    }

    if (a->n_rows != b->n_rows || a->n_cols != b->n_cols){
        printf("Tensor dimensions do not match for subtraction.\n");
        exit(0);
    }
    if (*output == NULL){
        Tensor_create(output, a->n_rows, a->n_cols);
    }
    else {
        // reallocate memory for output
        Tensor_realloc(*output, a->n_rows, a->n_cols);
    }
    
    for (size_t i = 0; i < a->n_rows; i++){
        for (size_t j = 0; j < a->n_cols; j++){
            Tensor_set(*output, i, j, Tensor_get(a, i, j) - Tensor_get(b, i, j));
        }
    }
    if (free) {Tensor_destroy(a); Tensor_destroy(b);}
};

void Tensor_print(Tensor* mat){
    for(size_t i = 0; i < mat->n_rows; i++){
        for(size_t j = 0; j < mat->n_cols; j++){
            printf("%f ", Tensor_get(mat, i, j));
        }
        printf("\n");
    }
};

void Tensor_multiply(Tensor* a, Tensor* b, Tensor** output, const unsigned char free){
    if (a == NULL){
        printf("Tensor a is pointing to an empty address\n.");
        return;
    }

    if (b == NULL){
        printf("Tensor b is pointing to an empty address\n.");
        return;
    }

    if (a->n_cols != b->n_rows){
        printf("Tensor dimensions do not match for multiplication.\n");
        exit(0);
    }
    
    // reallocate memory for output
    if (*output == NULL){
        Tensor_create(output, a->n_rows, b->n_cols);
    }
    else {
        Tensor_realloc(*output, a->n_rows, b->n_cols);    
    } 
    
    for (size_t i = 0; i < a->n_rows; i++){
        for (size_t j = 0; j < b->n_cols; j++){
            double sum = 0.0;
            for (size_t k = 0; k < a->n_cols; k++){
                sum += Tensor_get(a, i, k) * Tensor_get(b, k, j);
            }
            Tensor_set(*output, i, j, sum);
        }
    }

    if (free){ Tensor_destroy(a); Tensor_destroy(b);}
};

void Tensor_abs(Tensor* X){
    for (size_t i = 0; i < X->n_rows; i++){
        for (size_t j = 0; j < X->n_cols; j++){
            Tensor_set(X, i, j, abs(Tensor_get(X, i, j)));
        }
    }
};

Tensor* create_identity_Tensor(const size_t n){
    Tensor* mat = NULL;
    Tensor_create(&mat, n, n);
    for (size_t i = 0; i < n; i++){
        Tensor_set(mat, i, i, 1);
    }
    return mat;
};
Tensor* Tensor_copy(Tensor* mat){
    Tensor* copy = NULL;
    Tensor_create(&copy, mat->n_rows, mat->n_cols);
    memcpy(copy->data, mat->data, mat->n_rows * mat->n_cols * sizeof(double));
    return copy;
};

double Tensor_froebenius_norm(Tensor* mat){
    double sum = 0.0;
    for (size_t i = 0; i < mat->n_rows; i++){
        for (size_t j = 0; j < mat->n_cols; j++){
            sum += Tensor_get(mat, i, j) * Tensor_get(mat, i, j);
        }
    }
    return sqrt(sum);
};

void Tensor_sqrt(Tensor* X){
    for (size_t i = 0; i < X->n_rows; i++){
        for (size_t j = 0; j < X->n_cols; j++){
            Tensor_set(X, i, j, sqrt(Tensor_get(X, i, j)));
        }
    }
};

/* double calculate_inverse_tolerance(Tensor* A_T, Tensor* x){
    Tensor* prod = Tensor_multiply(A_T, x, 0);
    Tensor_print(x);
    Tensor* identity = create_identity_Tensor(A_T->n_rows);
    Tensor* diff = Tensor_subtract(prod, identity, 1);
    double tol = Tensor_froebenius_norm(diff);
    Tensor_destroy(&diff);
    return tol;
} */

/* Tensor* Tensor_inverse_newton(Tensor* mat, size_t max_iter, double tol, unsigned char free){
    Tensor* X = Tensor_create(mat->n_rows, mat->n_cols);

    // Initialize x_prev with 1.0s
    printf("Initializing Tensor with 1.0s.\n");
    for (size_t i = 0; i < X->n_rows; i++){
        for (size_t j = 0; j < X->n_cols; j++){
            Tensor_set(X, i, j, 1.0f);
        }
    }
 
    size_t iter = 0;
    Tensor* X_T = Tensor_transpose(mat);
    Tensor* scaled_identity = scalar_product(create_identity_Tensor(mat->n_cols), 2.0f, 1);
    printf("INITIAL X_T_n_cols: %lu, X_n_rows: %lu\n", X_T->n_cols, X->n_rows);
    while (iter <= max_iter && calculate_inverse_tolerance(X_T, X) > tol){
        printf("Loop %lu.\n", iter);
        printf("X_T_n_cols: %lu, X_n_rows: %lu\n", X_T->n_cols, X->n_rows);

        X = Tensor_multiply(X, Tensor_subtract(scaled_identity, Tensor_multiply(X_T, X, 0), 0), 0);
        iter++;
    }

    Tensor_destroy(&scaled_identity);
    Tensor_destroy(&X_T);
    if (free) Tensor_destroy(&mat);

    return X;
};
 */

Tensor* Tensor_create_from_array(const size_t n_rows, const size_t n_cols, const double (*arr)[n_cols]){
    if (arr == NULL){
        printf("Array to be converted to Tensor points to an empty address.\n");
        exit(0);
    }
    Tensor* mat = NULL;
    Tensor_create(&mat, n_rows, n_cols);
    for (size_t i = 0; i < n_rows; i++){
        for (size_t j = 0; j < n_cols; j++){
            Tensor_set(mat, i, j, arr[i][j]);
        }
    }



    return mat;
};
Tensor* tensor_new(const size_t n_rows, const size_t n_cols){
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL){
        printf("Failed to allocate memory for Tensor.\n");
        exit(1);
    }
    tensor->self = tensor;
    tensor->nodes = (ADNode**)malloc(n_rows * n_cols * sizeof(ADNode*));
    tensor->n_rows = n_rows;
    tensor->n_cols = n_cols;

    // Set methods
    tensor->realloc = tensor_realloc;
    tensor->destroy = tensor_destroy;
    tensor->set = tensor_set;
    tensor->get = tensor_get;   

    return tensor
};

#endif // __TENSOR_H__