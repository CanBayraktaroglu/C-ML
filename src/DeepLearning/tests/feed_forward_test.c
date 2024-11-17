#include "tensor.h"
#include "layers.h"
#include "compute_graph.h"

void main(void){

    ComputeGraph* graph = compute_graph_new(); 

    double arr[4][1] = {
        {1.0},
        {2.5},
        {6.0},
        {4.0},
    };
    
    Tensor* X = tensor_create_from_array(4, 1, arr);

    printf("Initializing Feed Forward Layer.\n");
    Layer* layer = init_layer(FEED_FORWARD, 1, 4, tensor_relu_inplace);
    
    printf("Forward Pass.\n");
    layer->forward(layer, X);

    printf("Printing resulting Tensor \n");
    X->print_val(X);
    
    printf("Building the graph..\n");
    graph_build(graph, X->get_node(X, 0, 0));

    printf("Propagating back..\n");
    graph->propagate_back(graph);

    X->detach(X);
    layer->destroy(layer);
    printf("Num nodes in graph: %lu\n", graph->num_nodes);
    graph->destroy(graph);


};