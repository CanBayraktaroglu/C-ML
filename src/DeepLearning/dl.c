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
    FeedForwardLayer* layer_0 = create_feed_forward_layer(0, 4, 0);
    FeedForwardLayer* layer_1 = create_feed_forward_layer(4, 1, 0);


    printf("Passing layer 0. \n");
    feed_forward_pass(layer_0, _X); // N_0 x 1
    printf("Row num: %lu, Col num: %lu \n.", _X->n_rows, _X->n_cols);
    matrix_print(_X);

    printf("Passing layer 1. \n");
    feed_forward_pass(layer_1, _X);
    printf("Row num: %lu, Col num: %lu \n.", _X->n_rows, _X->n_cols);
    matrix_print(_X);

    destroy_feed_forward_layer(layer_0);
    destroy_feed_forward_layer(layer_1);
    matrix_destroy(&X);
    matrix_destroy(&_X);
};