#ifndef __DL_H__
#define __DL_H__

#include <stdlib.h>	
#include <string.h>
#include "utils.h"
#include "matrix.h"
#include <math.h>
#include <assert.h>

// Loss Functions
void mean_squared_loss(Matrix* prediction, Matrix* label, Matrix** loss){
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
    matrix_sqrt(_loss);
    matrix_destroy(*loss);
    free(*loss);
    matrix_destroy(loss_T);
    free(loss_T);
    *loss = _loss;    
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

void backward_relu(Matrix* Z, Matrix** da_dz){
    if (*da_dz == NULL){
        matrix_create(da_dz, Z->n_rows, Z->n_cols);
    }

    for (size_t i = 0; i < Z->n_rows; i++){
        for (size_t j = 0; j < Z->n_cols; j++){
            if (matrix_get(Z, i, j) <= 0) continue;
            matrix_set(*da_dz, i, j, 1.0);
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

void backward_sigmoid(Matrix* Z, Matrix** da_dz){
    if (*da_dz == NULL){
        matrix_create(da_dz, Z->n_rows, Z->n_cols);
    }

    // da/dz = sigmoid(Z) * (1-sigmoid(Z))
    Matrix* _Z = NULL;
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

void linear(Matrix* X){
};

typedef struct{
    void (*actfn)(Matrix* X);
}actfn_utils;

// FEED FORWARD NEURAL NETWORK
typedef struct {
    size_t next_num_neurons;
    size_t num_neurons;
    void (*act_fn)(Matrix* X);
    char act_fn_mapping;
    Matrix* a_prev;
    Matrix* weights;
    Matrix* biases;
    Matrix* da_dz;
    Matrix* grad_delta;
    Matrix* grad_W;
    Matrix* grad_b;
}FeedForwardLayer;

void set_da_dz_feed_forward_layer(FeedForwardLayer* layer, Matrix* z){
    // In the backpropagation method the upcoming gradients must be considered for the 
    // calculation of delta_L too

    Matrix* da_dz = NULL;    
    matrix_create(&da_dz, z->n_rows, z->n_cols);

    switch(layer->act_fn_mapping){
        case 0: // linear
            for (size_t i = 0; i < da_dz->n_rows; i++){
                for (size_t j = 0; j < da_dz->n_cols; j++){
                    matrix_set(da_dz, i, j, 1.0);
                }
            } 
            break;
        case 1: // relu            
            for (size_t i = 0; i < da_dz->n_rows; i++){
                for (size_t j = 0; j < da_dz->n_cols; j++){
                    if (matrix_get(z, i, j) <= 0) continue;
                    matrix_set(da_dz, i, j, 1.0);
                }
            }
            break;
        case 2: // sigmoid
            double z_i_j, d_a_d_z_i_j;
            for (size_t i = 0; i < da_dz->n_rows; i++){
                for (size_t j = 0; j < da_dz->n_cols; j++){
                    z_i_j = matrix_get(z, i, j);
                    d_a_d_z_i_j = z_i_j * (1.0 - z_i_j);
                    matrix_set(da_dz, i, j, d_a_d_z_i_j);
                }
            }
            break;
        case 3: // tanh
            printf("NOT IMPLEMENTED YET\n.");
            matrix_destroy(da_dz);
            free(da_dz);
            exit(0);
            break;
    }
    layer->da_dz = da_dz;
};

// Feed Forward Pass
void feed_forward_pass(FeedForwardLayer* layer, Matrix* X){
    layer->a_prev = X;
    Matrix* _X = matrix_copy(X);
    Matrix* W = matrix_copy(layer->weights);
    Matrix* b = matrix_copy(layer->biases);

    matrix_multiply(W, _X, &X, 0); // X = W*X | num_neurons x 1
    matrix_destroy(_X);
    free(_X);
    _X = matrix_copy(X);
    matrix_add(_X, b, &X, 0); // X = X + B | num_neurons x 1

    layer->act_fn(X);
    
    // store da_dz of the layer for backprop
    set_da_dz_feed_forward_layer(layer, X);

    matrix_destroy(W);
    free(W);
    matrix_destroy(b);
    free(b);
    matrix_destroy(_X);
    free(_X);
  
};

void backprop_feed_forward_layer(FeedForwardLayer* layer, Matrix* delta_grad_next){ 
    double delta_grad_j = 0.0;
    double delta_grad_val = 0.0;
    double da_dz_j = 0.0;
    double delta_k = 0.0;
    double w_k_j = 0.0;

    // Set grad_delta of the layer
    for (size_t j = 0; j < layer->grad_delta->n_rows; j++){        
        da_dz_j = matrix_get(layer->da_dz, j, 0);            
        delta_grad_val = 0.0;
        
        for (size_t k = 0; k < delta_grad_next->n_rows; k++){
            delta_k = matrix_get(delta_grad_next, k, 0);
            w_k_j = matrix_get(layer->weights, k, j);
            delta_grad_val += delta_k * w_k_j;
        }

        delta_grad_val *= da_dz_j;
        matrix_set(layer->grad_delta, j, 0, delta_grad_val);
    }

    // set gradient weights
    double dC_dw_j_k = 0.0;
    for (size_t j = 0; j < layer->grad_W->n_rows; j++){
        for (size_t k = 0; k < layer->grad_W->n_cols; k++){
            dC_dw_j_k = matrix_get(delta_grad_next, j, k) * matrix_get(layer->a_prev, k, 0);
            matrix_set(layer->grad_W, j, k, dC_dw_j_k);            
        }
    }

    // set bias gradients
    double dC_db_j = 0.0;
    for (size_t j = 0; j < layer->grad_b->n_rows; j++){
        dC_db_j = matrix_get(delta_grad_next, j, 0);
        matrix_set(layer->grad_b, j, 0, dC_db_j);
    }

};

void optimize_sequential_nn(Sequential_NN* model, Matrix* loss){
    double loss_val = matrix_get(loss, 0, 0);
    // TODO
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

    matrix_destroy(layer->grad_W);
    free(layer->grad_W);
    layer->grad_W = NULL;

    matrix_destroy(layer->grad_b);
    free(layer->grad_b);
    layer->grad_b = NULL;

    matrix_destroy(layer->da_dz);
    free(layer->da_dz);
    layer->da_dz = NULL;

    matrix_destroy(layer->grad_delta);
    free(layer->grad_delta);
    layer->grad_delta = NULL;
};

// Allocate memory for FFNN
FeedForwardLayer* create_feed_forward_layer(size_t next_num_neurons, size_t num_neurons, char act_fn_mapping){
    FeedForwardLayer* layer = (FeedForwardLayer*)malloc(sizeof(FeedForwardLayer));

    // Set the neuron numbers
    layer->next_num_neurons = next_num_neurons;
    layer->num_neurons = num_neurons;

    // initialize biases and weights on the heap
    layer->biases = NULL;
    matrix_create(&(layer->biases), next_num_neurons, 1); // num_neurons x 1
    layer->weights = NULL;
    matrix_create(&(layer->weights), next_num_neurons, num_neurons); // N{i} x N_{i-1}

    // Initialize weight_grad
    layer->grad_W = NULL;
    matrix_create(&(layer->grad_W), next_num_neurons, num_neurons);

    // Initialize bias_grad
    layer->grad_b = NULL;
    matrix_create(&(layer->grad_b), next_num_neurons, num_neurons);

    // Initialize delta_grad
    layer->grad_delta = NULL;
    matrix_create(&(layer->grad_delta), num_neurons, 1);
    
    // Initialize a_i
    //layer->da_dz = NULL;
    //matrix_create(&(layer->da_dz), next_num_neurons, 1);

    switch(act_fn_mapping){
        case 0:
            layer->act_fn = linear;
            break;
        case 1:
            layer->act_fn = relu;
            break;
        case 2:
            layer->act_fn = sigmoid;
            break;
        case 3:
            layer->act_fn = _tanh;
            break;
        default:
            printf("Selected mapping for the activation function does not exist.\n");
            destroy_feed_forward_layer(layer);
            exit(0);
    };
    return layer;
};

void init_feed_forward_layer(FeedForwardLayer** layer_dptr, size_t next_num_neurons, size_t num_neurons, char act_fn_mapping){
    // Set the neuron numbers
    (*layer_dptr)->next_num_neurons = next_num_neurons;
    (*layer_dptr)->num_neurons = num_neurons;

    // initialize biases on the heap
    (*layer_dptr)->biases = NULL;
    matrix_create(&((*layer_dptr)->biases), next_num_neurons, 1); // num_neurons x 1

    // Set biases to 1.0
    for (size_t i = 0; i < (*layer_dptr)->biases->n_rows; i++){
        for (size_t j = 0; j < (*layer_dptr)->biases->n_cols; j++){
            matrix_set((*layer_dptr)->biases, i, j, 1.0);
        }
    }

    // Initalize weights
    (*layer_dptr)->weights = NULL;
    matrix_create(&((*layer_dptr)->weights), next_num_neurons, num_neurons); // N{i+1} x N_{i}
    
    // Set weights to 1.0
    for (size_t i = 0; i < (*layer_dptr)->weights->n_rows; i++){
        for (size_t j = 0; j < (*layer_dptr)->weights->n_cols; j++){
            matrix_set((*layer_dptr)->weights, i, j, 1.0);
        }
    }

    // initalize weight gradients
    (*layer_dptr)->grad_W = NULL;
    matrix_create(&((*layer_dptr)->grad_W), next_num_neurons, num_neurons); // N{i+1} x N_{i}

    // Initialize bias_grad
    (*layer_dptr)->grad_b = NULL;
    matrix_create(&((*layer_dptr)->grad_b), next_num_neurons, num_neurons);

    // Initialize delta_grad
    (*layer_dptr)->grad_delta = NULL;
    matrix_create(&((*layer_dptr)->grad_delta), num_neurons, 1);
    
    // Initialize a_i
    //(*layer_dptr)->da_dz = NULL;
    //matrix_create(&((*layer_dptr)->da_dz), next_num_neurons, 1);

    // set act_fn_mapping
    (*layer_dptr)->act_fn_mapping = act_fn_mapping;

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

void add_feed_forward_layer(Sequential_NN* model_ptr, size_t output_size, size_t input_size, char act_fn_mapping){
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
        //matrix_print(x);
    }
};

#endif // __DL_H__