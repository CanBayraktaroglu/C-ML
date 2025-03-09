#ifndef __COMPUTE_GRAPH_H__
#define __COMPUTE_GRAPH_H__

#include "autodifferentation.h"
#include "optimizer.h"

// ADNODE GRAPH IMPLEMENTATION
// Graph Structure

typedef struct ComputeGraph{
    struct ComputeGraph* self;
    ADNode* head;
    ADNode** nodes;
    size_t num_nodes;
    size_t capacity;
    Optimizer* optimizer;

    void (*add_node)(struct ComputeGraph* self, ADNode* node, size_t idx);
    void (*destroy)(struct ComputeGraph* self);
    void (*sort)(struct ComputeGraph* self);
    void (*propagate_back)(struct ComputeGraph* self);
    void (*prune)(struct ComputeGraph* self);
    void (*optimize)(struct ComputeGraph* self);
    void (*build)(struct ComputeGraph* self, ADNode* output);

}ComputeGraph;  


// Graph Operations
void add_node_to_graph(ComputeGraph* self, ADNode* node, size_t idx){
    if (idx == self->capacity){
        self->capacity *= 2;
        self->nodes = (ADNode**)realloc(self->nodes, self->capacity * sizeof(ADNode*));
    }
    self->nodes[idx] = node;
    self->num_nodes++;

    // Update topology index
    node->idx = idx;
    if (self->optimizer){
        if (node->type == WEIGHT){
            // Update weight count
            self->optimizer->u.adam.weight_count++;
            node->topology_idx.weight_idx = self->optimizer->u.adam.weight_count - 1;
        }
    else if (node->type == BIAS){
            // Update bias count
            self->optimizer->u.adam.bias_count++;
            node->topology_idx.bias_idx = self->optimizer->u.adam.bias_count - 1;
        }
    }
};

void graph_prune(ComputeGraph* self){
    if (self){
        for (size_t i = 0; i < self->num_nodes; i++){
            ADNode* node = self->nodes[i];
            if (node->is_trainable) continue;
            node->destroy(node);
        }
        free(self->nodes);
    }
};

void compute_graph_destroy(ComputeGraph* self){
    if (self){
        for (size_t i = 0; i < self->num_nodes; i++){
            ADNode* node = self->nodes[i];
            node->destroy(node);
            self->nodes[i] = NULL;
        }
        free(self->nodes);
        self->nodes = NULL;
        dl_optimizer_destroy(self->optimizer);
        free(self);
    }
};

void dfs_sort(ADNode* node, ADNode** sorted, size_t* idx){
    if (node == NULL || node->visited) return;
    node->visited = 1;

    for (size_t i = 0; i < node->num_parents; i++){
        dfs_sort(node->parents[i], sorted, idx);
    }

    node->idx = (*idx)++; 
    sorted[node->idx] = node;
};

void bfs_sort(ADNode* node, ADNode** sorted, size_t* idx){
    if (node == NULL) return;

    // Create a queue for BFS
    ADNode** queue = (ADNode**)malloc(node->num_parents * sizeof(ADNode*));
    size_t front = 0, rear = 0;

    // Enqueue the starting node
    queue[rear++] = node;
    node->visited = 1;

    while (front < rear) {
        ADNode* current = queue[front++];
        current->idx = (*idx)--;
        sorted[current->idx] = current;

        // Enqueue all unvisited parents
        for (size_t i = 0; i < current->num_parents; i++) {
            if (!current->parents[i]->visited) {
                queue[rear++] = current->parents[i];
                current->parents[i]->visited = 1;
            }
        }
    }

    free(queue);
}

void graph_topological_sort(ComputeGraph* graph){
    ADNode** sorted = (ADNode**)malloc(graph->num_nodes * sizeof(ADNode*));
    size_t idx = graph->num_nodes - 1;

    // Set all nodes as unvisited
    for (size_t i = 0; i < graph->num_nodes; i++){
        graph->nodes[i]->visited = 0;
    }

    // Traverse and add nodes to corresponding places in sorted array
    for (size_t i = 0; i < graph->num_nodes; i++){
        if (!graph->nodes[i]->visited) bfs_sort(graph->nodes[i], sorted, &idx);
    }

    // Replace the original array with the sorted one
    free(graph->nodes);
    graph->nodes = sorted;
}

void dfs_traverse(ComputeGraph* graph, ADNode* node, size_t idx){
    if (graph == NULL) return;
    if (node == NULL || node->visited) return;
    
    node->visited = 1;
    add_node_to_graph(graph, node, idx);

    for (size_t i = 0; i < node->num_parents; i++){
        dfs_traverse(graph, node->parents[i], graph->num_nodes);
    }

};

void bfs_explore(ComputeGraph* graph, ADNode* node){
    if (graph == NULL) return;
    if (node == NULL) return;

    // Create a queue for BFS
    ADNode** queue = (ADNode**)malloc(node->num_parents * sizeof(ADNode*));
    size_t front = 0, rear = 0;
    size_t capacity = 10;

    // Enqueue the starting node
    queue[rear++] = node;
    node->visited = 1;

    while (front < rear) {
        ADNode* current = queue[front++];
        queue[front - 1] = NULL;
        add_node_to_graph(graph, current, current->idx);   

        // Enqueue all unvisited parents
        for (size_t i = 0; i < current->num_parents; i++) {
            if (!current->parents[i]->visited) {
                if (rear >= capacity){
                    capacity *= 2;
                    queue = (ADNode**)realloc(queue, capacity * sizeof(ADNode*));
                }
                
                queue[rear++] = current->parents[i];
                current->parents[i]->visited = 1;
            }
        }
    }

    free(queue);
}	

void dfs_backward(ComputeGraph* graph, ADNode* node){
    if (node == NULL || node->visited) return;
    node->visited = 1;
    if (node->backward){
        // Backpropagate
        node->backward(node);
        if (node->is_trainable && graph->optimizer){
            // Optimize
            Optimizer* optimizer = graph->optimizer;
            dl_optimizer_adam_update(optimizer, node);
        } 
    } 

    for (size_t i = 0; i < node->num_parents; i++){
        dfs_backward(graph, node->parents[i]);
    }

};

void bfs_backward(ComputeGraph* graph, ADNode* node){
    if(node == NULL) return;

    // Create a queue for BFS
    // *INEFFICIENT SOLUTION w.r.t heap memory complexity*
    ADNode** queue = (ADNode**)malloc(graph->num_nodes * sizeof(ADNode*));
    size_t front = 0, rear = 0;

    // Enqueue the starting node
    if (node->backward) node->backward(node);
    queue[rear++] = node;
    node->visited = 1;

    while (front < rear) {
        ADNode* current = queue[front++];
        // Enqueue all unvisited parents
        for (size_t i = 0; i < current->num_parents; i++) {
            if (!current->parents[i]->visited) {
                queue[rear++] = current->parents[i];
                current->parents[i]->visited = 1;
                if (current->parents[i]->backward) current->parents[i]->backward(current->parents[i]);
            }
        }
    }

    free(queue);   

};

void graph_propagate_back(ComputeGraph* self){
    // BFS more memory-effficient for skewed tree
    // DFS more memory-efficient for balanced tree
    // runtime complexity same for both O(V + E)
    // Set gradient of the output Node to 1
    self->head->data.grad = 1.0;
    self->optimizer->t++;

    // Set all nodes to unvisited
    printf("Setting all nodes to unvisited.\n");
    for (size_t i = 0; i < self->num_nodes; i++){
        self->nodes[i]->visited = 0;
    }

    // Traverse graph and propagate back
    printf("Backwarding graph.\n");
    dfs_backward(self, self->head);
    
};

/* void graph_optimize(ComputeGraph* self){
    self->optimizer->optimize(self->optimizer, self->nodes, self->num_nodes);
}; */

void graph_build(ComputeGraph* graph, ADNode* output){ 
    if (graph == NULL){
        printf("Graph is NULL\n");
        return;
    }

    if (output == NULL){
        printf("Output Node is NULL\n");
        return;
    }


    // Set head of graph
    graph->head = output;

    // Traverse graph of nodes
    dfs_traverse(graph, graph->head, 0);

};      

ComputeGraph* compute_graph_create(Optimizer* optimizer){
    ComputeGraph* graph = (ComputeGraph*)malloc(sizeof(ComputeGraph));
    graph->capacity = 10; // start with space for 10 Nodes
    graph->nodes = (ADNode**)malloc(graph->capacity * sizeof(ComputeGraph*));
    graph->num_nodes = 0;
    graph->self = graph;
    graph->optimizer = optimizer;
    // Set methods
    graph->add_node = add_node_to_graph;
    graph->destroy = compute_graph_destroy;
    graph->sort = graph_topological_sort;
    graph->propagate_back = graph_propagate_back;
    graph->prune = graph_prune;
    graph->build = graph_build;
    // graph->optimize = graph_optimize;

    return graph; 
};
#pragma endregion Computation Graph

#endif //  __COMPUTE_GRAPH_H__