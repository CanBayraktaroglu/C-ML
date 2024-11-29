#ifndef LOSS_H_
#define LOSS_H_

#include <stdlib.h>	
#include <string.h>
#include "utils.h"
#include "matrix.h"
#include <math.h>
#include "tensor.h"


typedef struct MSE_Loss{
    size_t num_samples;
    double loss;
}MSE_Loss;

#pragma region Loss Functions


// Loss Functions
void L2_loss(Matrix* prediction, Matrix* label, Matrix** loss){
    if (prediction->n_rows != label->n_rows || prediction->n_cols != label->n_cols){
        printf("Matrix sizes do not match\n.");
        matrix_destroy(prediction);
        matrix_destroy(label);
        free(prediction);
        prediction = NULL;
        free(label);
        label = NULL;
        exit(0);
    }

    matrix_subtract(prediction, label, loss, 0);
    Matrix* _loss = NULL;
    Matrix* loss_T = matrix_transpose(*loss);
    matrix_multiply(loss_T, *loss, &_loss, 0);
    matrix_destroy(*loss);
    free(*loss);
    matrix_destroy(loss_T);
    free(loss_T);
    *loss = _loss;    
};

Tensor* L2_loss_tensor(Tensor* prediction, Tensor* label){
    if (prediction->n_rows != label->n_rows || prediction->n_cols != label->n_cols){
        printf("Tensor sizes do not match\n.");
        exit(0);
    }
    Tensor* diff = tensor_subtract(prediction, label); 
    Tensor* diff_T = tensor_transpose(diff);
    Tensor* loss = tensor_dot_product(diff_T, diff);
    
    tensor_detach(diff);
    tensor_detach(diff_T);
    return loss;
};

MSE_Loss MSE_loss_new(){
    MSE_Loss mse_loss = {.loss=0.0, .num_samples = 0};  
    return mse_loss;
};

void MSE_loss(MSE_Loss* mse_loss, Tensor* prediction, Tensor* label){
    if (prediction->n_rows != label->n_rows || prediction->n_cols != label->n_cols){
        printf("Tensor sizes do not match\n.");
        exit(0);
    }

    if (mse_loss == NULL){
        printf("Provided MSE is NULL\n.");
        return;
    }
        
    double prev_loss = mse_loss->loss * mse_loss->num_samples;
    mse_loss->num_samples++;
    Tensor* prev_loss_tensor = tensor_new_init(1, 1, prev_loss);

    tensor_subtract_inplace(prediction, label);
    Tensor* diff_T = tensor_transpose(prediction);
    tensor_dot_product_reversed_order_inplace(prediction, diff_T);

    tensor_add_inplace(prediction, prev_loss_tensor);

    tensor_scalar_product_inplace(prediction, 1.0 / mse_loss->num_samples);

    mse_loss->loss = tensor_get_val(prediction, 0, 0);

    tensor_detach(prev_loss_tensor);
    tensor_detach(diff_T);
    

};
 
void backward_L2_loss(Matrix* a_out, Matrix* y, Matrix** dC_da_out){
    if (a_out == NULL || y == NULL){
        printf("aout or y pointing to NULL address\n.");
        exit(0);
    }
    
    Matrix* mat = NULL;
    matrix_subtract(a_out, y, &mat, 0);
    scalar_product(mat, 2.0, dC_da_out, 0);

    matrix_destroy(mat);
    free(mat);

};

void L1_loss(Matrix* prediction, Matrix* label, Matrix** loss){
    if (prediction->n_rows != label->n_rows || prediction->n_cols != label->n_cols){
        printf("Matrix sizes do not match\n.");
        matrix_destroy(prediction);
        matrix_destroy(label);
        free(prediction);
        prediction = NULL;
        free(label);
        label = NULL;
        exit(0);
    }

    matrix_subtract(prediction, label, loss, 0);
    Matrix* mat = NULL;
    matrix_create(&mat, label->n_rows, 1);
    Matrix* output = NULL;
    matrix_multiply(*loss, mat, &output, 0);

    // Free allocated heap memory
    matrix_destroy(mat);
    free(mat);
    mat = NULL;

    matrix_destroy(*loss);
    free(*loss);
    *loss = output;

};
#pragma endregion Loss Functions

#endif // LOSS_H_