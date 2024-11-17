#include "models.h"
#include "layers.h"
#include "optimizer.h"
#include "matrix.h"
#include "loss.h"

void main(void){

    double arr[4][1] = {
        {1.0},
        {2.5},
        {6.0},
        {4.0},
    };
    
    Matrix* X = matrix_create_from_array(4, 1, arr);
    Matrix* _X = matrix_copy(X);
    Sequential_NN_* sequential_nn = NULL;
    init_sequential_nn_(&sequential_nn, 4, 3, 2);
    
    if (sequential_nn == NULL){
        printf("Layers point to null address.\n");
        exit(0);
    }

    // Stack layers on the sequential Model
        add_feed_forward_layer_(sequential_nn, sequential_nn->hidden_size, sequential_nn->input_size, 0);
        add_feed_forward_layer_(sequential_nn, sequential_nn->hidden_size, sequential_nn->hidden_size, 0);
        add_feed_forward_layer_(sequential_nn, sequential_nn->output_size, sequential_nn->hidden_size, 0);
        print_sequential_nn_(sequential_nn);

    // Forward Pass 
        forward_sequential_nn_(sequential_nn, _X);
    
    // Optimizer
        Adam_Optimizer* optimizer = NULL;
        printf("INITIALIZE OPTIMIZER.\n");
        init_Adam_optimizer(&optimizer, 0.004f, 0.5f, 0.9f, 0.9f, 0.000001f, sequential_nn->layers, sequential_nn->num_layers);

    // data
        double label[2][1] = {30, 20};
        Matrix* y = matrix_create_from_array(2, 1, label);

    // Loss Calculation
        Matrix* loss = NULL;
        L2_loss(_X, y, &loss);
        matrix_print(loss);

    // Propagate back the gradients top-to-bottom 
        printf("BACKPROPAGATION.\n");
        backpropagate_sequential_nn_(sequential_nn, _X, y, 0);

    // Optimize
        printf("OPTIMIZATION.\n");
        optimize_adam(optimizer, sequential_nn->layers);
        
    // Free allocated dynamic memory
        destroy_sequential_nn_(sequential_nn);
        free(sequential_nn);
        destroy_adam_optimizer(optimizer);
        free(optimizer);        
        matrix_destroy(X);
        free(X);
        matrix_destroy(y);
        free(y);
        matrix_destroy(loss);
        free(loss);
        matrix_destroy(_X);
        free(_X);
};