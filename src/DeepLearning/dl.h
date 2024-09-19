#ifndef __DL_H__
#define __DL_H__

#include <stdlib.h>	
#include <string.h>
#include "utils.h"
#include "matrix.h"
#include <math.h>
#include <assert.h>


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
 
void backward_L2_loss(Matrix* a_out, Matrix* y, Matrix** dC_da_out){
    if (a_out == NULL || y == NULL){
        printf("aout or y pointing to NULL address\n.");
        exit(0);
    }
    
    Matrix* mat = NULL;
    matrix_subtract(a_out, y, &mat, 0);
    scalar_product(mat, 2.0, dC_da_out, 0);

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

#pragma region Activation Functions

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

void linear(Matrix* X){
};
#pragma endregion Activation Functions


#pragma region Feed Forward Layer 

// FEED FORWARD Layer
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
#pragma endregion Feed Forward Layer

#pragma region Sequential Neural Network 
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
    }
};

void optimize_sequential_nn(Sequential_NN* model, Matrix* a_out, Matrix* y, char loss_fn){
    
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

    for (size_t i = model->num_layers - 1; i >= 0; i--){
        Layer* layer_ptr = model->layers + i;
        switch(layer_ptr->type){
            case FEED_FORWARD:
                FeedForwardLayer* ff_layer_ptr = layer_ptr->layer.ff_layer;

                // Calculate Gradients
                backprop_feed_forward_layer(ff_layer_ptr, delta_grad_next);
                delta_grad_next = ff_layer_ptr->grad_delta;

                // Update Weights
                Matrix* layer_W = ff_layer_ptr->weights;
                for (size_t i = 0; i < layer_W->n_rows; i++){
                    for (size_t j = 0; j < layer_W->n_cols; j++){
                        double w_i_j = matrix_get(layer_W, i, j);
                        w_i_j -= 0.0;// OPTIMIZATION;

                    }
                }

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

#pragma region Optimizer

#pragma region Adam
/*
Adam Optimizer 

Adaptive Moment Estimation is an algorithm for optimization technique for gradient descent. 
The method is really efficient when working with large problem involving a lot of data or parameters.
It requires less memory and is efficient. Intuitively, it is a combination of the ‘gradient descent 
    with momentum’ algorithm and the ‘RMSP’ algorithm. 
A combination of two gradient descent methodoligies

Momentum: 

This algorithm is used to accelerate the gradient descent algorithm by taking into consideration 
the ‘exponentially weighted average’ of the gradients. Using averages makes the algorithm converge
towards the minima in a faster pace. ​
                                w_t+1 = w_t - alpha * m_t
                          where m_t = beta_1 * m_t-1 + (1-beta_1)*[delta_W[i][t]] (1)


Root Mean Square Propagation:

Root mean square prop or RMSprop is an adaptive learning algorithm that tries to improve AdaGrad.
Instead of taking the cumulative sum of squared gradients like in AdaGrad, 
    it takes the ‘exponential moving average’.
                                w_t+1 = w_t - (alpha_t)/(v_t + epsilon)^(0.5) * [delta_W[i][t]]
                          where v_t = beta_2*v_t-1 + (1-beta_2)*[delta_W[i][t]]^2 (2)

Since mt and vt have both initialized as 0 (based on the eq. (1) and (2)),
it is observed that they gain a tendency to be ‘biased towards 0’ as both β1 & β2 ≈ 1.
This Optimizer fixes this problem by computing ‘bias-corrected’ mt and vt.
This is also done to control the weights while reaching the global minimum to prevent 
    high oscillations when near it. 

The formulas used are:
                                m_dach_t = m_t/(1-beta_1^t)
                                v_dach_t = v_t/(1-beta_2^t)

        --> w_t+1 = w_t - m_dach_t*(alpha/sqrt(v_dach_t + epsilon))
*/

typedef struct{
    double learning_rate;
    double alpha;
    double beta_1;
    double beta_2;
    size_t num_layers;
    Matrix* m_w_ptr;
    Matrix* m_b_ptr;
    Matrix* v_w_ptr;  
    Matrix* v_b_ptr;
}Adam_Optimizer;

void init_Adam_optimizer(Adam_Optimizer** optimizer_dptr, double lr, double alpha, double beta_1, double beta_2, Layer* layers, size_t num_layers){
    if (*optimizer_dptr == NULL){
        *optimizer_dptr = (Adam_Optimizer*)malloc(sizeof(Adam_Optimizer)); 
    }

    (*optimizer_dptr)->alpha = alpha;
    (*optimizer_dptr)->beta_1 = beta_1;
    (*optimizer_dptr)->beta_2 = beta_2;
    (*optimizer_dptr)->learning_rate = lr;
    (*optimizer_dptr)->num_layers = num_layers;

    // Allocate space for gradients for weights and biases in each layer
    (*optimizer_dptr)->m_w_ptr = (Matrix*)malloc(num_layers*sizeof(Matrix));
    (*optimizer_dptr)->m_b_ptr = (Matrix*)malloc(num_layers*sizeof(Matrix)); 
    (*optimizer_dptr)->v_w_ptr = (Matrix*)malloc(num_layers*sizeof(Matrix));
    (*optimizer_dptr)->v_b_ptr = (Matrix*)malloc(num_layers*sizeof(Matrix));
    
    // Initialize
    for (size_t i = 0; i < num_layers; i++){
        Layer* layer_ptr = layers + i;
        switch(layer_ptr->type){
            case FEED_FORWARD:
                FeedForwardLayer* ff_layer_ptr = layer_ptr->layer.ff_layer;
                
                //set rows and cols
                matrix_realloc((*optimizer_dptr)->m_w_ptr + i, ff_layer_ptr->grad_W->n_rows, ff_layer_ptr->grad_W->n_cols);
                matrix_realloc((*optimizer_dptr)->v_w_ptr + i, ff_layer_ptr->grad_W->n_rows, ff_layer_ptr->grad_W->n_cols);
                matrix_realloc((*optimizer_dptr)->m_b_ptr + i, ff_layer_ptr->grad_b->n_rows, ff_layer_ptr->grad_b->n_cols);
                matrix_realloc((*optimizer_dptr)->v_b_ptr + i, ff_layer_ptr->grad_b->n_rows, ff_layer_ptr->grad_b->n_cols);

            default:
                printf("Provided layer type not supported.\n");
                break;     

        }   
    }

};

void optimize_adam(Adam_Optimizer* optimizer, Layer* layers){
    for (size_t i = 0; i < optimizer->num_layers; i++){
        Layer* layer_ptr = layers + i;
        switch(layer_ptr->type){
            case FEED_FORWARD:
                FeedForwardLayer* ff_layer_ptr = layer_ptr->layer.ff_layer;
                for (size_t j = 0; j < ff_layer_ptr->grad_W->n_rows; j++){
                    for (size_t k =0; k < ff_layer_ptr->grad_W->n_cols; k++){
                        // TODO
                    }
                }
        }
    }   
}

#pragma endregion Adam

typedef enum{
    Adam,
    BASE,
}OptimizerType;

typedef union{
    Adam_Optimizer adam_optimizer;    
}OptimizerUnion;

typedef struct
{
    OptimizerType optimizer_type;
    OptimizerUnion optimizer;

}Optimizer;


#pragma endregion Optimizer

#endif // __DL_H__