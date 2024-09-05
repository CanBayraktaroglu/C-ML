#ifndef __DL_H__
#define __DL_H__

#include <stdlib.h>	
#include <string.h>
#include "utils.h"
#include "matrix.h"
#include <math.h>

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


// FEED FORWARD NEURAL NETWORK
typedef struct {
    size_t prev_layer_num_neurons;
    size_t num_neurons;
    void (*act_fn)(Matrix* X);
    Matrix* weights;
    Matrix* biases;
}FeedForwardLayer;


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

// Garbage Collector Funcs
void destroy_feed_forward_layer(FeedForwardLayer* layer){
    if (layer == NULL) return;
    matrix_destroy(&layer->weights);
    matrix_destroy(&layer->biases);
    free(layer);
    layer = NULL;
};

// Allocate memory for FFNN
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

// Sequential NN
typedef struct{
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    size_t num_layers;
    char layer_type;
    char act_fn;
    void* layers;
}Sequential_NN;

// Allocate memory on the heap for Sequential NN
Sequential_NN* create_Sequential_NN(size_t input_size, size_t hidden_size, size_t output_size, size_t num_layers, char layer_type, char act_fn){
    Sequential_NN* sequential_nn = (Sequential_NN*)calloc(1, sizeof(Sequential_NN));

    sequential_nn->input_size = input_size;
    sequential_nn->hidden_size = hidden_size;
    sequential_nn->output_size = output_size;
    sequential_nn->num_layers = num_layers;
    sequential_nn->layer_type = layer_type;
    sequential_nn->act_fn = act_fn;

    return sequential_nn;
};

void initialize_sequential_nn(Sequential_NN* model){
    if (model->num_layers < 2){
        printf("At least 2 layers must be present\n.");
        free(model);
        model = NULL;
        exit(0);
    }

    void** layer_ptr; 
    switch(model->layer_type){
        case 0: // Feed Forward
            model->layers = (FeedForwardLayer*)calloc(model->num_layers, sizeof(FeedForwardLayer));
            layer_ptr = (FeedForwardLayer**)&model->layers;

            // Input layer
            *layer_ptr = create_feed_forward_layer(0, model->input_size, model->act_fn);

            // Hidden layers
            for (size_t i = 1; i < model->num_layers - 1; i++){
                size_t prev_layer_size = (*(layer_ptr) + i - 1)->num_neurons;
                *(layer_ptr + i) = create_feed_forward_layer(prev_layer_size, model->hidden_size, model->act_fn);                   
            }

            // Output Layer
            *(layer_ptr + model->num_layers - 1) = create_feed_forward_layer(model->hidden_size, model->output_size, model->act_fn);
            break;
    
        default:
            printf("Chosen layer type is not supported\n.");
            exit(0);   
    }
};

void destroy_sequential_nn(Sequential_NN* model){
    return;
};

#endif // __DL_H__