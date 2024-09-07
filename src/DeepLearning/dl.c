#include "dl.h"
#include "matrix.h"

void main(void){
    // TODO Weight-layer assignment is wrong
    double arr[4][1] = {
        {1},
        {2},
        {3},
        {4},
    };
    
    Matrix* X = matrix_create_from_array(4, 1, arr);
    Matrix* _X = matrix_copy(X);
    Sequential_NN* sequential_nn = NULL;
    init_sequential_nn(&sequential_nn, 4, 6, 1);
    
    if (sequential_nn == NULL){
        printf("Layers point to null address.\n");
        exit(0);
    }
    add_feed_forward_layer(sequential_nn, 0, sequential_nn->input_size, 0);
    add_feed_forward_layer(sequential_nn, sequential_nn->input_size, sequential_nn->hidden_size, 1);
    
    
    // Free allocated dynamic memory
    destroy_sequential_nn(sequential_nn);        
    free(sequential_nn);
    matrix_destroy(X);
    free(X);
    matrix_destroy(_X);
    free(_X);
};