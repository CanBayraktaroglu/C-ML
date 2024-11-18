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
    
    TensorDataset* dataset = tensor_dataset_new(10);

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

    tensor_dataset_add_X(dataset, X);
    tensor_dataset_add_y(dataset, y);

    // Add layers to the model
    add_feed_forward_layer(model, 20, 4, tensor_relu_inplace);
    add_feed_forward_layer(model, 1, 20, tensor_relu_inplace);

    // Initialize optimizer
    Adam_Optimizer* optimizer = init_Adam_optimizer(0.004f, 0.5f, 0.9f, 0.9f, 0.000001f, model->layers, model->num_layers);
    
    print_sequential_nn(model);

    printf("Model Params: %lu\n", model->num_params);

    sequential_nn_train(model, dataset, 5, optimizer);

    /* printf("Forward Pass.\n");
    forward_sequential_nn(model, X);
    printf("--------------------\n");
    tensor_print_val(X);
    printf("--------------------\n");
    Tensor* loss = L2_loss_tensor(X, y);
    printf("Loss: %f\n", tensor_get_val(loss, 0, 0));

    printf("Building the graph..\n");
    graph_build(graph, tensor_get_node(loss, 0, 0)); */

    // Print values of all layer weights and biases
    //printf("Printing values of layer params.\n");
    //sequential_nn_print_params(model);

    //printf("Propagating back..\n");
    //printf("%f \n",graph->head->data.value);
    /* graph_propagate_back(graph); */
    
    // Print grad and value of layer weights
    //printf("Printing gradients of layer params after BACK PROPAGATION.\n");
    //sequential_nn_print_grads(model);

   /*  printf("Optimizing..\n");
    optimizer_step(optimizer, model->layers);
    printf("Optimization done.\n"); */

    // Print grad and value of layer weights
    //printf("Printing values of layer params after OPTIMIZATION.\n");
    //sequential_nn_print_params(model);

    destroy_sequential_nn(model);
    graph_destroy(graph);
    destroy_adam(optimizer);
    tensor_dataset_destroy(dataset);
    printf("Done.\n");
};