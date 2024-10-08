#ifndef __MODELS_H__
#define __MODELS_H__

#include <stdlib.h>	
#include <string.h>
#include "utils.h"
#include "matrix.h"
#include <math.h>
#include <assert.h>
#include "act_fn..h"
#include "layers.h"
#include "loss.h"

#pragma region Sequential Neural Network 

// Sequential NN
typedef struct Sequential_NN{
    struct Sequential_NN* self;
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    size_t num_layers;
    Layer* layers;
}Sequential_NN;

// Allocate memory on the heap for Sequential NN
Sequential_NN* create_Sequential_NN(const size_t input_size, const size_t hidden_size, const size_t output_size, const size_t num_layers, const char layer_type, const char act_fn){
    Sequential_NN* sequential_nn = (Sequential_NN*)calloc(1, sizeof(Sequential_NN));

    sequential_nn->self = sequential_nn;
    sequential_nn->input_size = input_size;
    sequential_nn->hidden_size = hidden_size;
    sequential_nn->output_size = output_size;
    sequential_nn->num_layers = num_layers;

    return sequential_nn;
};

void init_sequential_nn(Sequential_NN** model, const size_t input_size, const size_t hidden_size, const size_t output_size) {
    
    *model = (Sequential_NN*)calloc(1, sizeof(Sequential_NN)); 
    Sequential_NN* model_ptr = *model;
    
    if (*model == NULL) {
        printf("Failed to allocate memory for model\n");
        exit(1);
    }

    // Set the attributes
    model_ptr->self = *model;
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

void add_feed_forward_layer(Sequential_NN* model_ptr, size_t output_size, size_t input_size, const char act_fn_mapping){
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
    init_feed_forward_layer(&layer_ptr, output_size, input_size, act_fn_mapping);
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
    }
};

void backpropagate_sequential_nn(Sequential_NN* model, Matrix* a_out, Matrix* y, const char loss_fn){
    
    Matrix* dC_da_out = NULL;
    switch(loss_fn){
        case 0: //L2 loss
            backward_L2_loss(a_out, y, &dC_da_out);
            break;
        default:
            printf("Selected loss function is not supported.\n");
            exit(0);
    }
    
    Matrix* delta_grad_next = dC_da_out;

    for (int i = model->num_layers - 1; i >= 0; i--){
        Layer* layer_ptr = model->layers + i;
        switch(layer_ptr->type){
            case FEED_FORWARD:
                FeedForwardLayer* ff_layer_ptr = layer_ptr->layer.ff_layer;

                // Calculate Gradients
                backprop_feed_forward_layer(ff_layer_ptr, delta_grad_next);
                delta_grad_next = ff_layer_ptr->grad_delta;

                break;
            default:
                printf("Layer type not supported.\n");
                exit(0);
        }
    }

    matrix_destroy(dC_da_out);
    free(dC_da_out);



};

#pragma endregion Sequential Neural Network



#endif // __DL_H__