#include "tensor.h"
#include "layers.h"
#include "compute_graph.h"
#include "models.h"
#include "optimizer.h"  

void main(void){

    // Initialize sequential model
    Sequential_NN* model = init_sequential_nn();

    // Initialize graph
    ComputeGraph* graph = compute_graph_new(); 
    
    // Initialize generic input tensor 
    double arr[4][1] = {
        {1.0},
        {2.5},
        {6.0},
        {4.0},
    };

    double arr_2[1][1] = {
        {1.0},
    };
    
    Tensor* X = tensor_create_from_array(4, 1, arr);
    Tensor* y = tensor_create_from_array(1, 1, arr_2);

    //printf("Initializing Feed Forward Layer.\n");
    add_feed_forward_layer(model, 200000, 4, tensor_relu_inplace);
    add_feed_forward_layer(model, 1, 200000, tensor_relu_inplace);

    // Initialize optimizer
    Adam_Optimizer* optimizer = init_Adam_optimizer(0.004f, 0.5f, 0.9f, 0.9f, 0.000001f, model->layers, model->num_layers);
    
    printf("Printing Sequential NN\n");
    print_sequential_nn(model);

    printf("Model Params: %lu\n", model->num_params);

    printf("Forward Pass.\n");
    //layer->forward(layer, X);
    forward_sequential_nn(model, X);
    printf("--------------------\n");
    tensor_print_val(X);
    printf("--------------------\n");
    Tensor* loss = L2_loss_tensor(X, y);
    printf("Loss: %f\n", tensor_get_val(loss, 0, 0));

    printf("Building the graph..\n");
    graph_build(graph, tensor_get_node(loss, 0, 0));

    // Print values of all layer weights and biases
    //printf("Printing values of layer params.\n");
    //sequential_nn_print_params(model);

    //printf("Propagating back..\n");
    //printf("%f \n",graph->head->data.value);
    graph_propagate_back(graph);
    
    // Print grad and value of layer weights
    //printf("Printing gradients of layer params after BACK PROPAGATION.\n");
    //sequential_nn_print_grads(model);

    printf("Optimizing..\n");
    optimize_adam(optimizer, model->layers);
    printf("Optimization done.\n");

    // Print grad and value of layer weights
    //printf("Printing values of layer params after OPTIMIZATION.\n");
    //sequential_nn_print_params(model);

    tensor_detach(X);
    tensor_detach(y);
    tensor_detach(loss);
    destroy_sequential_nn(model);
    graph_destroy(graph);
    destroy_adam(optimizer);

};