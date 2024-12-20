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

    Tensor* B = tensor_sigmoid(A);
    printf("Sigmoid\n"); 
    B->print_val(B);

    printf("Exp\n");
    Tensor* C = tensor_exp(B);

    printf("Transpose\n");
    Tensor* D = B->transpose(B);
    D->print_val(D);

    printf("L2 Loss\n");
    Tensor* loss = L2_loss_tensor(B, C);
    
    printf("Building graph.\n");
    graph_build(compute_graph, loss->get_node(loss, 0, 0));

    printf("Loss: %f\n", loss->get_val(loss, 0, 0));
    printf("Graph num nodes: %lu\n", compute_graph->num_nodes);

    //Backward
    printf("Propagating back.\n");
    graph_propagate_back(compute_graph);
    
    printf("Printing gradients of C.\n");
    C->print_grad(C);

    // TODO OPTIMIZATION

    printf("Printing gradients of B.\n");
    B->print_grad(B);

    printf("Printing gradients of A.\n");
    A->print_grad(A);

    printf("Detaching tensors from nodes.\n");
    A->detach(A);
    B->detach(B);
    C->detach(C);

    loss->detach(loss);
    D->detach(D);

    printf("Destroying graph.\n"); 
    compute_graph->destroy(compute_graph);
};
