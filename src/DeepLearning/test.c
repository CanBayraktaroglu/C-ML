#include "compute_graph.h"
#include "tensor.h"
#include "loss.h"

void main(void){
    ComputeGraph* compute_graph = graph_new();

    double arr_1[4][1] = {
        {1.0},
        {2.5},
        {6.0},
        {4.0},
    };

    Tensor* A = tensor_create_from_array(4, 1, arr_1);

    Tensor* B = A->sigmoid(A);
    printf("Sigmoid\n"); 
    B->print_val(B);

    const double C = B->froebenius_norm(B);
    printf("Froebenius Norm: %f\n", C);

    Tensor* D = B->transpose(B);
    printf("Transpose\n");
    D->print_val(D);

    printf("L2 Loss\n");
    Tensor* loss = L2_loss_tensor(A, B);
    
    // TODO BUILD GRAPH
    printf("Building graph.\n");
    compute_graph->build(compute_graph, loss->get_node(loss, 0, 0));

    printf("Loss: %f\n", loss->get_val(loss, 0, 0));
    printf("Graph num nodes: %lu\n", compute_graph->num_nodes);

    printf("Destroying graph.\n"); 
    compute_graph->destroy(compute_graph);

    printf("Destroying tensors.\n");
    A->free(A);
    B->free(B);
    loss->free(loss);
    D->free(D);
};
