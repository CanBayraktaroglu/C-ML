#include "compute_graph.h"
#include "tensor.h"
#include "loss.h"

void main(void){
    Optimizer* optimizer = dl_optimizer_adam_create(0.01, 0.9, 0.999, 1e-8, 1e-8);
    ComputeGraph* compute_graph = compute_graph_create(optimizer);

    double arr_1[4][1] = {
        {1.0},
        {2.5},
        {6.0},
        {4.0},
    };

    Tensor* A = tensor_create_from_array(4, 1, arr_1, INPUT);

    Tensor* B = tensor_sigmoid(A, WEIGHT);
    // Set all nodes to trainable to test optimizer
    for (size_t i = 0; i < B->n_rows; i++){
        for (size_t j = 0; j < B->n_cols; j++){
            ADNode* node = B->get_node(B, i, j);
            node->is_trainable = 1;
        }
    }
    printf("Sigmoid\n"); 
    //B->print_val(B);

    printf("Exp\n");
    Tensor* C = tensor_exp(B, POST_ACTIVATION);

    printf("Transpose\n");
    Tensor* D = tensor_transpose(B);
    //D->print_val(D);

    printf("L2 Loss\n");
    Tensor* loss = L2_loss_tensor(B, C);
    printf("Loss: %f\n", loss->get_val(loss, 0, 0));
    
    printf("Building graph.\n");
    graph_build(compute_graph, loss->get_node(loss, 0, 0));

    printf("Graph num nodes: %lu\n", compute_graph->num_nodes);

    //Backward
    printf("Propagating back.\n");
    graph_propagate_back(compute_graph);
    
    printf("Printing gradients of C.\n");
    C->print_grad(C);    

    printf("Printing gradients of B.\n");
    B->print_grad(B);

    printf("Printing gradients of A.\n");
    A->print_grad(A);

    printf("Print optimizer values.\n");
    for (size_t i = 0; i < optimizer->u.adam.weight_count; i++){
        printf("Weight m_w %zu: %f\n", i, optimizer->u.adam.w_dptr[i]->m_w);
        printf("Weight v_w %zu: %f\n", i, optimizer->u.adam.w_dptr[i]->v_w);
    }
    for (size_t i = 0; i < optimizer->u.adam.bias_count; i++){
        printf("Bias m_b %zu: %f\n", i, optimizer->u.adam.b_dptr[i]->m_b);
        printf("Bias v_b %zu: %f\n", i, optimizer->u.adam.b_dptr[i]->v_b);
    }

    printf("Detaching tensors from nodes.\n");
    A->detach(A);
    B->detach(B);
    C->detach(C);

    loss->detach(loss);
    D->detach(D);

    printf("Destroying graph.\n"); 
    compute_graph_destroy(compute_graph);
};
