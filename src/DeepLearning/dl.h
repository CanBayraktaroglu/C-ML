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
    Matrix* (*act_fn)(Matrix* X, size_t prev_layer_num_neurons, size_t num_neurons);
    Matrix* weights;
    Matrix* biases;
}FeedForwardLayer;

// ACTIVATION FUNCTIONS
Matrix* relu(Matrix* X, size_t prev_layer_num_neurons, size_t num_neurons){
    if (prev_layer_num_neurons){
        X->n_rows = num_neurons;
        X->n_cols = prev_layer_num_neurons;
        for (size_t i = 0; i < num_neurons; i++){
            for (size_t j = 0; j < prev_layer_num_neurons; j++){
                if (matrix_get(X, i, j) >= 0) continue;
                matrix_set(X, i, j, 0.0f);
            }
        }
    }
    return X;
};

Matrix* sigmoid(Matrix* X, size_t prev_layer_num_neurons, size_t num_neurons){
    if (prev_layer_num_neurons){
        X->n_rows = num_neurons;
        X->n_cols = prev_layer_num_neurons;
        for (size_t i = 0; i < num_neurons; i++){
            for (size_t j = 0; j < prev_layer_num_neurons; j++){
                double x = matrix_get(X, i, j);
                matrix_set(X, i, j, 1/(1 + exp(-x)));
            }
        }
    }
    return X;
};

Matrix* _tanh(Matrix* X, size_t prev_layer_num_neurons, size_t num_neurons){
    if (prev_layer_num_neurons){
        X->n_rows = num_neurons;
        X->n_cols = prev_layer_num_neurons;
        for (size_t i = 0; i < num_neurons; i++){
            for (size_t j = 0; j < prev_layer_num_neurons; j++){
                double x = matrix_get(X, i, j);
                matrix_set(X, i, j, exp(2*x - 1)/exp(2*x + 1));
            }
        }
    }
    return X;
};

// Feed Forward Pass
Matrix* feed_forward_pass(FeedForwardLayer* layer , Matrix* X){
    if (layer->prev_layer_num_neurons){
        X = matrix_multiply(layer->weights, X, 0); // X = W*X | num_neurons x 1
        X = matrix_add(X, layer->biases, 0); // X = X + B | num_neurons x 1 
        X = layer->act_fn(X, layer->prev_layer_num_neurons, layer->num_neurons);
    }
    
    return X;
};

void destroy_feed_forward_layer(FeedForwardLayer* layer){
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