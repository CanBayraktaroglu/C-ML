#ifndef __MODELS_H__
#define __MODELS_H__

#include <stdlib.h>	
#include <string.h>
#include "utils.h"
#include "matrix.h"
#include <math.h>
#include <assert.h>
#include "layers.h"
#include "loss.h"
#include "compute_graph.h"  

typedef struct Sequential_NN{
    size_t num_layers;
    size_t num_params;
    Layer** layers;
}Sequential_NN;

Sequential_NN* init_sequential_nn(){
    Sequential_NN* model = (Sequential_NN*)malloc(sizeof(Sequential_NN));
    model->num_layers = 0;
    model->num_params = 0;
    model->layers = NULL;
    return model;
};

void add_feed_forward_layer(Sequential_NN* model, size_t output_size, size_t input_size, void (*act_fn)(Tensor* X)){
    if (model == NULL){
        printf("Passed model pointer is NULL.\n");
        exit(0);
    }
    
    // Increment number of layers
    model->num_layers++;
    model->num_params += output_size * input_size + output_size;

    if (model->layers == NULL){
        model->layers = (Layer**)malloc(sizeof(Layer*));
    }else{
        model->layers = (Layer**)realloc(model->layers, model->num_layers * sizeof(Layer*));
    }

    // Initialize feed forward neural layer
    Layer* layer = init_layer(FEED_FORWARD, output_size, input_size, act_fn);
    *(model->layers + model->num_layers - 1) = layer;
};

void destroy_sequential_nn(Sequential_NN* model){
    if (model == NULL) return;
    
    for (size_t i = 0; i < model->num_layers; i++){
        Layer* layer = *(model->layers + i);
        layer->destroy(layer);
        *(model->layers + i) = NULL;    
    }    

    free(model->layers);
    model->layers = NULL;
    free(model);

};

void print_sequential_nn(Sequential_NN* model){
    printf("LAYERS:\n");
    for (size_t i = 0; i < model->num_layers; i++){
        Layer* layer = *(model->layers + i);
        printf("    Idx: %lu ", i);
        switch(layer->type){
            case FEED_FORWARD:
                printf("Type: FEED FORWARD, # out_neurons: %lu, # in_neurons: %lu", layer->num_neurons, layer->num_features);
                break;
            default:
                printf("TYPE NOT SUPPORTED.");
                break;
        }
        printf("\n");
    }
};

void sequential_nn_print_params(Sequential_NN* model){
    for (size_t i = 0; i < model->num_layers; i++){
        Layer* layer = *(model->layers + i);
        switch(layer->type){
            case FEED_FORWARD:
                printf("--------------------\n");
                printf("Layer %lu\n", i);
                printf("Weights:\n");
                tensor_print_val(layer->layer.ff_layer->weights);
                printf("Biases:\n");
                tensor_print_val(layer->layer.ff_layer->biases);
                printf("--------------------\n");
                break;
            default:
                printf("Layer type not supported.\n");
                break;
        }
    }
};

void sequential_nn_print_grads(Sequential_NN* model){
    for (size_t i = 0; i < model->num_layers; i++){
        Layer* layer = *(model->layers + i);
        switch(layer->type){
            case FEED_FORWARD:
                printf("--------------------\n");
                printf("Layer %lu\n", i);
                printf("Weights:\n");
                tensor_print_grad(layer->layer.ff_layer->weights);
                printf("Biases:\n");
                tensor_print_grad(layer->layer.ff_layer->biases);
                printf("--------------------\n");
                break;
            default:
                printf("Layer type not supported.\n");
                break;
        }
    }
};

void forward_sequential_nn(Sequential_NN* model, Tensor* X){
    for (size_t i = 0; i < model->num_layers; i++){
        Layer* layer = *(model->layers + i);
        layer->forward(layer, X);
    }
};

void sequential_nn_train(Sequential_NN* model, TensorDataset* dataset, size_t epochs, Adam_Optimizer* optimizer){

    if (model == NULL){
        printf("Model is pointing to NULL.\n");
        exit(0);
    }

    if (optimizer == NULL){
        printf("Optimizer is pointing to NULL.\n");
        exit(0);
    }

    if (dataset == NULL){
        printf("Dataset is pointing to NULL.\n");
        exit(0);
    }

    assert(dataset->n_x == dataset->n_y);
    
    for (size_t epoch = 0; epoch < epochs; epoch++){
        printf("Epoch: %lu\n", epoch);
        for (size_t i = 0; i < dataset->n_x; i++){
            ComputeGraph* graph = compute_graph_new();
            Tensor* _x = tensor_dataset_get_X(dataset, i);
            
            if (_x == NULL){
                printf("X is pointing to NULL.\n");
                exit(0);
            }
            
            Tensor* _y = tensor_dataset_get_y(dataset, i);            
            if (_y == NULL){
                printf("Y is pointing to NULL.\n");
                exit(0);
            }
            
            Tensor* x = tensor_copy(_x);
            if (x == NULL){
                printf("Failed to copy tensor x.\n");
                exit(0);
            }

            Tensor* y = tensor_copy(_y);            
            if (y == NULL){
                printf("Failed to copy tensor y.\n");
                exit(0);
            }
            
            printf("Forward Pass.\n");
            forward_sequential_nn(model, x);

            if (x == NULL){
                printf("X is pointing to NULL.\n");
                exit(0);
            }

            Tensor* loss = L2_loss_tensor(x, y);
            printf("Loss: %f\n", tensor_get_val(loss, 0, 0));
            printf("--------------------\n");
            printf("Graph Num Nodes %lu\n", graph->num_nodes);
            graph_build(graph, loss->get_node(loss, 0, 0));
            printf("--------------------\n");
            printf("Graph Num Nodes %lu\n", graph->num_nodes);

            graph_propagate_back(graph);

            optimizer_step(optimizer, model->layers);

            printf("Printing values of layer params.\n");
            sequential_nn_print_params(model);

            tensor_detach(loss);
            tensor_detach(x);
            tensor_detach(y);

            if (epoch != (epochs - 1) || i != (dataset->n_x - 1)) {
                graph_prune(graph);

            }   
            
        }
    }
};

#pragma region Sequential Neural Network Layerswise 

// Sequential NN
typedef struct Sequential_NN_{
    struct Sequential_NN_* self;
    size_t input_size;
    size_t hidden_size;
    size_t output_size;
    size_t num_layers;
    Layer_* layers;
}Sequential_NN_;

// Allocate memory on the heap for Sequential NN
Sequential_NN_* create_Sequential_NN(const size_t input_size, const size_t hidden_size, const size_t output_size, const size_t num_layers, const char layer_type, const char act_fn){
    Sequential_NN_* sequential_nn = (Sequential_NN_*)calloc(1, sizeof(Sequential_NN_));

    sequential_nn->self = sequential_nn;
    sequential_nn->input_size = input_size;
    sequential_nn->hidden_size = hidden_size;
    sequential_nn->output_size = output_size;
    sequential_nn->num_layers = num_layers;

    return sequential_nn;
};

void init_sequential_nn_(Sequential_NN_** model, const size_t input_size, const size_t hidden_size, const size_t output_size) {
    
    *model = (Sequential_NN_*)calloc(1, sizeof(Sequential_NN_)); 
    Sequential_NN_* model_ptr = *model;
    
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

void destroy_sequential_nn_(Sequential_NN_* model_ptr){
    if (model_ptr == NULL) return;
    
    for (size_t i = 0; i < model_ptr->num_layers; i++){
        Layer_* layer_ptr = model_ptr->layers + i;
        
        switch(layer_ptr->type){
            case FEED_FORWARD:
                FeedForwardLayer_* ff_layer_ptr = layer_ptr->layer.ff_layer;
                destroy_feed_forward_layer_(ff_layer_ptr);
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

void add_feed_forward_layer_(Sequential_NN_* model_ptr, size_t output_size, size_t input_size, const char act_fn_mapping){
    if (model_ptr == NULL){
        printf("Passed model pointer is NULL.\n");
        exit(0);
    }
    // Increment number of layers
    model_ptr->num_layers++;

    if (model_ptr->layers == NULL){
        model_ptr->layers = (Layer_*)malloc(sizeof(Layer_));
    }else{
        model_ptr->layers = (Layer_*)realloc(model_ptr->layers, model_ptr->num_layers * sizeof(Layer_));
    }

    // Initialize feed forward neural layer
    (model_ptr->layers + model_ptr->num_layers - 1)->type = FEED_FORWARD;
    (model_ptr->layers + model_ptr->num_layers - 1)->layer.ff_layer = (FeedForwardLayer_*)malloc(sizeof(FeedForwardLayer_));
    FeedForwardLayer_* layer_ptr = (model_ptr->layers + model_ptr->num_layers - 1)->layer.ff_layer;
    
    if (layer_ptr == NULL){
        printf("Layer ptr points to NULL.\n Feed forward layer cannot be added.\n");
        exit(0);
    } 
    
    init_feed_forward_layer_(&layer_ptr, output_size, input_size, act_fn_mapping);
};

void print_sequential_nn_(Sequential_NN_* model_ptr){
    printf("LAYERS:\n");
    for (size_t i = 0; i < model_ptr->num_layers; i++){
        
        Layer_* layer_ptr = model_ptr->layers + i;
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

void forward_sequential_nn_(Sequential_NN_* model_ptr, Matrix* x){
    for (size_t i = 0; i < model_ptr->num_layers; i++){
        Layer_* layer_ptr = (model_ptr->layers + i);
        switch (layer_ptr->type){
            case FEED_FORWARD:
                FeedForwardLayer_* ff_layer_ptr = layer_ptr->layer.ff_layer;
                feed_forward_pass(ff_layer_ptr, x);
                break;
            default:
                printf("Layer Type not supported.");
                break;
        }
    }
};

void backpropagate_sequential_nn_(Sequential_NN_* model, Matrix* a_out, Matrix* y, const char loss_fn){
    
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
        Layer_* layer_ptr = model->layers + i;
        switch(layer_ptr->type){
            case FEED_FORWARD:
                FeedForwardLayer_* ff_layer_ptr = layer_ptr->layer.ff_layer;

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