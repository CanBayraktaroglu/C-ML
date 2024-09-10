#include "dl.h"
#include "matrix.h"

void main(void){
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
        printf("Layers point to null address.\n");
        exit(0);
    }

    // Stack layers on the sequential Model
    add_feed_forward_layer(sequential_nn, 0, sequential_nn->input_size, 0);
    add_feed_forward_layer(sequential_nn, sequential_nn->input_size, sequential_nn->hidden_size, 0);
    add_feed_forward_layer(sequential_nn, sequential_nn->hidden_size, sequential_nn->hidden_size, 0);
    add_feed_forward_layer(sequential_nn, sequential_nn->hidden_size, sequential_nn->output_size, 0);
    print_sequential_nn(sequential_nn);

    // Forward Pass 
    forward_sequential_nn(sequential_nn, _X);
    matrix_print(_X);
    double label[2][1] = {540.0, 500.0};
    Matrix* mat = matrix_create_from_array(2, 1, label);
    Matrix* loss = NULL;
    mean_squared_loss(_X, mat, &loss);
    matrix_print(loss);
    
    // Free allocated dynamic memory
    destroy_sequential_nn(sequential_nn);        
    free(sequential_nn);
    matrix_destroy(X);
    free(X);
    matrix_destroy(mat);
    free(mat);
    matrix_destroy(loss);
    free(loss);
    matrix_destroy(_X);
    free(_X);
};