#ifndef __DL_H__
#define __DL_H__

#include <stdlib.h>	
#include <string.h>
#include "utils.h"
#include "matrix.h"
#include <math.h>
#include <assert.h>

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
    matrix_destroy(layer->weights);
    free(layer->weights);
    layer->weights = NULL;
    matrix_destroy(layer->biases);
    free(layer->biases);
    layer->biases = NULL;
};

// Allocate memory for FFNN
FeedForwardLayer* create_feed_forward_layer(size_t prev_layer_num_neurons, size_t num_neurons, char act_fn_mapping){
    FeedForwardLayer* layer = (FeedForwardLayer*)malloc(sizeof(FeedForwardLayer));

    // Set the neuron numbers
    layer->prev_layer_num_neurons = prev_layer_num_neurons;
    layer->num_neurons = num_neurons;

    // initialize biases and weights on the heap
    Matrix* b = NULL;
    matrix_create(&b, num_neurons, 1); // num_neurons x 1
    layer->biases = b;

    Matrix* W = NULL;
    matrix_create(&W, num_neurons, prev_layer_num_neurons); // N{i} x N_{i-1}
    layer->weights = W;

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

void init_feed_forward_layer(FeedForwardLayer** layer_dptr, size_t prev_layer_num_neurons, size_t num_neurons, char act_fn_mapping){
    // Set the neuron numbers
    (*layer_dptr)->prev_layer_num_neurons = prev_layer_num_neurons;
    (*layer_dptr)->num_neurons = num_neurons;

    // initialize biases and weights on the heap
    (*layer_dptr)->biases = NULL;
    matrix_create(&((*layer_dptr)->biases), num_neurons, 1); // num_neurons x 1

    // Set biases to 1.0
    for (size_t i = 0; i < (*layer_dptr)->biases->n_rows; i++){
        for (size_t j = 0; j < (*layer_dptr)->biases->n_cols; j++){
            matrix_set((*layer_dptr)->biases, i, j, 1.0);
        }
    }

    (*layer_dptr)->weights = NULL;
    matrix_create(&((*layer_dptr)->weights), num_neurons, prev_layer_num_neurons); // N{i} x N_{i-1}
    
    // Set weights to 1.0
    for (size_t i = 0; i < (*layer_dptr)->weights->n_rows; i++){
        for (size_t j = 0; j < (*layer_dptr)->weights->n_cols; j++){
            matrix_set((*layer_dptr)->weights, i, j, 1.0);
        }
    }

    switch(act_fn_mapping){
        case 0:
            (*layer_dptr)->act_fn = relu;
            break;
        case 1:
            (*layer_dptr)->act_fn = sigmoid;
            break;
        case 2:
            (*layer_dptr)->act_fn = _tanh;
            break;
        default:
            printf("Selected mapping for the activation function does not exist.\n");
            exit(0);
    };
};

typedef enum {
    FEED_FORWARD,
    CONVOLUTIONAL,
    //add more types as needed
}LayerType;

typedef union {
    FeedForwardLayer* ff_layer;
    void* layer;
}LayerUnion;

typedef struct {
    LayerType type;
    LayerUnion layer;
}Layer;

// Sequential NN
typedef struct{
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    size_t num_layers;
    Layer* layers;
}Sequential_NN;

// Allocate memory on the heap for Sequential NN
Sequential_NN* create_Sequential_NN(size_t input_size, size_t hidden_size, size_t output_size, size_t num_layers, char layer_type, char act_fn){
    Sequential_NN* sequential_nn = (Sequential_NN*)calloc(1, sizeof(Sequential_NN));

    sequential_nn->input_size = input_size;
    sequential_nn->hidden_size = hidden_size;
    sequential_nn->output_size = output_size;
    sequential_nn->num_layers = num_layers;

    return sequential_nn;
};

void init_sequential_nn(Sequential_NN** model, size_t input_size, size_t hidden_size, size_t output_size) {
    
    *model = (Sequential_NN*)calloc(1, sizeof(Sequential_NN)); 
    Sequential_NN* model_ptr = *model;
    
    if (*model == NULL) {
        printf("Failed to allocate memory for model\n");
        exit(1);
    }

    // Set the attributes
    model_ptr->input_size = input_size;
    model_ptr->hidden_size = hidden_size;
    model_ptr->output_size = output_size;
    model_ptr->num_layers = 0;
    
    // Initialize layers as NULL
    model_ptr->layers = NULL;

    printf("Sequential NN INITIALIZED.\n");
};

void destroy_sequential_nn(Sequential_NN* model_ptr){
    if (model_ptr == NULL) return;
    
    for (size_t i = 0; i < model_ptr->num_layers; i++){
        Layer* layer_ptr = model_ptr->layers + i;
        
        switch(layer_ptr->type){
            case FEED_FORWARD:
                FeedForwardLayer* ff_layer_ptr = layer_ptr->layer.ff_layer;
                destroy_feed_forward_layer(ff_layer_ptr);
                free(ff_layer_ptr);
                ff_layer_ptr = NULL;
                break;
            default:
                printf("Specified layer type is not supported");
                break;
        }        
    }    

    free(model_ptr->layers);
    model_ptr->layers = NULL;
};

void add_feed_forward_layer(Sequential_NN* model_ptr, size_t prev_layer_num_neurons, size_t num_neurons, char act_fn_mapping){
    if (model_ptr == NULL){
        printf("Passed model pointer is NULL.\n");
        exit(0);
    }
    // Increment number of layers
    model_ptr->num_layers++;

    if (model_ptr->layers == NULL){
        model_ptr->layers = (Layer*)malloc(sizeof(Layer));
    }else{
        model_ptr->layers = (Layer*)realloc(model_ptr->layers, model_ptr->num_layers * sizeof(Layer));
    }

    // Initialize feed forward neural layer
    (model_ptr->layers + model_ptr->num_layers - 1)->type = FEED_FORWARD;
    (model_ptr->layers + model_ptr->num_layers - 1)->layer.ff_layer = (FeedForwardLayer*)malloc(sizeof(FeedForwardLayer));
    FeedForwardLayer* layer_ptr = (model_ptr->layers + model_ptr->num_layers - 1)->layer.ff_layer;
    
    if (layer_ptr == NULL){
        printf("Layer ptr points to NULL.\n Feed forward layer cannot be added.\n");
        exit(0);
    } 
    init_feed_forward_layer(&layer_ptr, prev_layer_num_neurons, num_neurons, act_fn_mapping);
};

void print_sequential_nn(Sequential_NN* model_ptr){
    printf("LAYERS:\n");
    for (size_t i = 0; i < model_ptr->num_layers; i++){
        
        Layer* layer_ptr = model_ptr->layers + i;
        printf("    Idx: %lu ", i);
        
        switch(layer_ptr->type){
            case FEED_FORWARD:
                printf("Type: FEED FORWARD, # Neurons: %lu ", layer_ptr->layer.ff_layer->num_neurons);
                break;
            default:
                printf("TYPE NOT SUPPORTED.");
                break;
        }
        
        printf("\n");
    }
};

void forward_sequential_nn(Sequential_NN* model_ptr, Matrix* x){
    for (size_t i = 0; i < model_ptr->num_layers; i++){
        Layer* layer_ptr = (model_ptr->layers + i);
        switch (layer_ptr->type){
            case FEED_FORWARD:
                FeedForwardLayer* ff_layer_ptr = layer_ptr->layer.ff_layer;
                feed_forward_pass(ff_layer_ptr, x);
                break;
            default:
                printf("Layer Type not supported.");
                break;
        }
        //matrix_print(x);
    }
};

#endif // __DL_H__