#include "compute_graph.h"
#include "tensor.h"
#include "loss.h"

void main(void){
    ComputeGraph* graph = graph_new();

    double arr_1[1][1] = {{10.0}};

    Tensor* A = tensor_create_from_array(1, 1, arr_1);
    A->print_val(A);

    double arr_2[4][1] = {
        {1.0},
        {2.5},
        {6.0},
        {4.0},
    };

    Tensor* B = tensor_create_from_array(4, 1, arr_2);

    Tensor* C = tensor_create_from_array(4, 1, arr_2);

    printf("L2 Loss\n");
    Tensor* loss = L2_loss_tensor(C, B);
    
    // TODO BUILD GRAPH
    printf("Building graph.\n");
    graph->build(graph, loss->get_node(loss, 0, 0));

    printf("Graph num nodes: %lu\n", graph->num_nodes);

    printf("Destroying graph.\n"); 
    graph->destroy(graph);

    printf("Destroying tensors.\n");
    A->destroy(A);
    printf("A destroyed.\n");
    B->free(B);
    printf("B destroyed.\n");
    loss->free(loss);
    printf("Loss Destroyed.\n");
    C->destroy(C);
    printf("C destroyed.\n");
};
