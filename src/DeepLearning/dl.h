#ifndef __DL_H__
#define __DL_H__

#include <stdlib.h>	
#include <string.h>
#include "utils.h"
#include "matrix.h"
#include <math.h>

typedef struct {
    size_t prev_layer_num_neurons;
    size_t num_neurons;
    void (*act_fn)(Matrix* X);
    Matrix* weights;
    Matrix* biases;
}FeedForwardLayer;

// ACTIVATION FUNCTIONS
void relu(Matrix* X){
    if (X->n_cols){        
        for (size_t i = 0; i < X->n_rows; i++){
            for (size_t j = 0; j < X->n_cols; j++){
                if (matrix_get(X, i, j) >= 0) continue;
                matrix_set(X, i, j, 0.0f);
            }
        }

    }
};

void sigmoid(Matrix* X){
    if (X->n_cols){
  
        for (size_t i = 0; i < X->n_rows; i++){
            for (size_t j = 0; j < X->n_cols; j++){
                double x = matrix_get(X, i, j);
                matrix_set(X, i, j, 1/(1 + exp(-x)));
            }
        }
    }
};

void _tanh(Matrix* X){
    if (X->n_cols){
        for (size_t i = 0; i < X->n_rows; i++){
            for (size_t j = 0; j < X->n_cols; j++){
                double x = matrix_get(X, i, j);
                matrix_set(X, i, j, exp(2*x - 1)/exp(2*x + 1));
            }
        }
    }
};

// Feed Forward Pass
void feed_forward_pass(FeedForwardLayer* layer , Matrix* X){
    if (layer->prev_layer_num_neurons){
        Matrix* _X = matrix_copy(X);
        Matrix* W = matrix_copy(layer->weights);
        Matrix* b = matrix_copy(layer->biases);

        matrix_multiply(W, _X, X, 1); // X = W*X | num_neurons x 1

        _X = matrix_copy(X);
        matrix_add(_X, b, X, 1); // X = X + B | num_neurons x 1

        layer->act_fn(X);
    }  
};

void destroy_feed_forward_layer(FeedForwardLayer* layer){
    if (layer == NULL) return;
    matrix_destroy(&layer->weights);
    matrix_destroy(&layer->biases);
    free(layer);
    layer = NULL;
};

FeedForwardLayer* create_feed_forward_layer(size_t prev_layer_num_neurons, size_t num_neurons, char act_fn_mapping){
    FeedForwardLayer* layer = (FeedForwardLayer*)malloc(sizeof(FeedForwardLayer));

    // Set the neuron numbers
    layer->prev_layer_num_neurons = prev_layer_num_neurons;
    layer->num_neurons = num_neurons;

    // initialize biases and weights on the heap
    layer->biases = matrix_create(num_neurons, 1); // num_neurons x 1
    layer->weights = matrix_create(num_neurons, prev_layer_num_neurons); // N{i} x N_{i-1}

    switch(act_fn_mapping){
        case 0:
            layer->act_fn = relu;
            break;
        case 1:
            layer->act_fn = sigmoid;
            break;
        case 2:
            layer->act_fn = _tanh;
            break;
        default:
            printf("Selected mapping for the activation function does not exist.\n");
            destroy_feed_forward_layer(layer);
            exit(0);
    };


    return layer;
};

#endif // __DL_H__