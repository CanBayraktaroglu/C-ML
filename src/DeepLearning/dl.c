#include "dl.h"
#include "matrix.h"
#include "logger.h"

void main(void){
    // Logger
    Logger logger = init_Logger();

    double arr[4][1] = {
        {1.0},
        {2.5},
        {6.0},
        {4.0},
    };
    
    Matrix* X = matrix_create_from_array(4, 1, arr);
    Matrix* _X = matrix_copy(X);
    Sequential_NN* sequential_nn = NULL;
    init_sequential_nn(&sequential_nn, 4, 6, 2);
    
    if (sequential_nn == NULL){
        logger.err("Layers point to null address.");
        exit(0);
    }

    // Stack layers on the sequential Model
        add_feed_forward_layer(sequential_nn, sequential_nn->hidden_size, sequential_nn->input_size, 0);
        add_feed_forward_layer(sequential_nn, sequential_nn->hidden_size, sequential_nn->hidden_size, 0);
        add_feed_forward_layer(sequential_nn, sequential_nn->hidden_size, sequential_nn->hidden_size, 0);
        add_feed_forward_layer(sequential_nn, sequential_nn->output_size, sequential_nn->hidden_size, 0);
        print_sequential_nn(sequential_nn);

    // Forward Pass 
        forward_sequential_nn(sequential_nn, _X);
    
    // Optimizer
        Adam_Optimizer* optimizer = NULL;
        init_Adam_optimizer(&optimizer, 0.004f, 0.5, 0.9f, 0.9f, 0.000001f, sequential_nn->layers, sequential_nn->num_layers);

    // data
        double label[2][1] = {540.0, 500.0};
        Matrix* mat = matrix_create_from_array(2, 1, label);

    // Loss Calculation
        Matrix* loss = NULL;
        L2_loss(_X, mat, &loss);
        matrix_print(loss);

    // Propagate back the gradients top-to-bottom 
        backprop_feed_forward_layer(sequential_nn->layers, loss);
    
    // Optimize
        optimize_adam(optimizer, sequential_nn->layers);


    // Free allocated dynamic memory
        destroy_sequential_nn(sequential_nn);
        free(sequential_nn);
        destroy_adam_optimizer(optimizer);
        free(optimizer);        
        matrix_destroy(X);
        free(X);
        matrix_destroy(mat);
        free(mat);
        matrix_destroy(loss);
        free(loss);
        matrix_destroy(_X);
        free(_X);
};