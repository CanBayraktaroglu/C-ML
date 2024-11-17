#include "tensor.h"
#include "layers.h"

void main(void){

    double arr[4][1] = {
        {1.0},
        {2.5},
        {6.0},
        {4.0},
    };
    
    printf("Creating Tensor from array.\n");
    Tensor* X = tensor_create_from_array(4, 1, arr);

    printf("Initializing Feed Forward Layer.\n");
    Layer* layer = init_layer(FEED_FORWARD, 1, 4, tensor_relu_inplace);
    
    printf("Forward Pass.\n");
    layer->forward(layer, X);

    printf("Printing resulting Tensor \n");
    X->print_val(X);

};