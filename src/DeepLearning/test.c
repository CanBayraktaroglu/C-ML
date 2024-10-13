#include "compute_graph.h"
#include "tensor.h"
#include "loss.h"

void main(void){
    ComputeGraph* compute_graph = graph_new();

    double arr_1[1][1] = {{10.0}};

    Tensor* A = tensor_create_from_array(1, 1, arr_1);
    Tensor*a = A->relu(A);
    
    a->print_val(a);

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
    compute_graph->build(compute_graph, loss->get_node(loss, 0, 0));

    printf("Graph num nodes: %lu\n", compute_graph->num_nodes);

    printf("Destroying graph.\n"); 
    compute_graph->destroy(compute_graph);

    printf("Destroying tensors.\n");
    A->destroy(A);
    a->destroy(a);
    B->free(B);
    loss->free(loss);
    C->free(C);
};
