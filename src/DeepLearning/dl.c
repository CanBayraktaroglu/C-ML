#include "dl.h"
#include "matrix.h"

void main(void){

    Matrix* X = matrix_create(5, 1);
    FeedForwardLayer* layer_0 = create_feed_forward_layer(0, 5, 0);
    FeedForwardLayer* layer_1 = create_feed_forward_layer(5, 1, 0);

    printf("Passing layer 0. \n");
    X = feed_forward_pass(layer_0, X); // N_0 x 1
    printf("Row num: %lu, Col num: %lu \n.", X->n_rows, X->n_cols);
    matrix_print(X);

    printf("Passing layer 1. \n");
    X = feed_forward_pass(layer_1, X);
    printf("Row num: %lu, Col num: %lu \n.", X->n_rows, X->n_cols);
    matrix_print(X);

    destroy_feed_forward_layer(layer_0);
    destroy_feed_forward_layer(layer_1);
    matrix_destroy(&X);
};