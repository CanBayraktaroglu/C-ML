#include "compute_graph.h"
#include "tensor.h"

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

    Tensor* C = B->dot_product(B, A);

    C->print_val(C);

    // TODO BUILD GRAPH
    

    printf("Destroying graph.\n"); 
    graph->free(graph);

    printf("Destroying tensors.\n");
    A->destroy(A);
    printf("A destroyed.\n");
    B->destroy(B);
    printf("B destroyed.\n");
    C->destroy(C);
    printf("C destroyed.\n");
};
