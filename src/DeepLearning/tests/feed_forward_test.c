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

    double arr_2[3][1] = {
        {1.0},
        {2.5},
        {6.0},
    };
    
    Tensor* X = tensor_create_from_array(4, 1, arr);
    Tensor* y = tensor_create_from_array(1, 1, arr_2);

    printf("Initializing Feed Forward Layer.\n");
    //Layer* layer = init_layer(FEED_FORWARD, 1, 4, tensor_relu_inplace);
    add_feed_forward_layer(model, 3, 4, tensor_relu_inplace);
    Layer* layer_0 = *(model->layers + 0);
    add_feed_forward_layer(model, 1, 3, tensor_relu_inplace);
    Layer* layer_1 = *(model->layers + 1);

    // Initialize optimizer
    Adam_Optimizer* optimizer = init_Adam_optimizer(0.004f, 0.5f, 0.9f, 0.9f, 0.000001f, model->layers, model->num_layers);
    
    printf("Printing Sequential NN\n");
    print_sequential_nn(model);

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

    // Print grad and value of layer weights
    tensor_print_val(layer_0->layer.ff_layer->weights);
    printf("--------------------\n");
    tensor_print_val(layer_1->layer.ff_layer->weights); 
    printf("--------------------\n");
    tensor_print_val(layer_0->layer.ff_layer->biases);
    printf("--------------------\n");
    tensor_print_val(layer_1->layer.ff_layer->biases);

    printf("Propagating back..\n");
    printf("%f \n",graph->head->data.value);
    graph_propagate_back(graph);
    
    // Print grad and value of layer weights
    printf("Printing gradients of layer params after BACK PROPAGATION.\n");

    tensor_print_grad(layer_0->layer.ff_layer->weights);
    printf("--------------------\n");
    tensor_print_grad(layer_0->layer.ff_layer->biases);
    printf("--------------------\n");	
    tensor_print_grad(layer_1->layer.ff_layer->weights);
    printf("--------------------\n");
    tensor_print_grad(layer_1->layer.ff_layer->biases);

    printf("Optimizing..\n");
    optimize_adam(optimizer, model->layers);
    printf("Optimization done.\n");

    // Print grad and value of layer weights
    printf("Printing values of layer params after OPTIMIZATION.\n");
    tensor_print_val(layer_0->layer.ff_layer->weights);
    printf("--------------------\n");
    tensor_print_val(layer_1->layer.ff_layer->weights); 
    printf("--------------------\n");

    tensor_detach(X);
    tensor_detach(y);
    tensor_detach(loss);
    destroy_sequential_nn(model);
    //printf("Number of nodes in graph: %lu\n", graph->num_nodes);
    graph_destroy(graph);
    destroy_adam(optimizer);

};