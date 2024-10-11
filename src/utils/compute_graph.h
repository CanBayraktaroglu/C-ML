#ifndef __COMPUTE_GRAPH_H__
#define __COMPUTE_GRAPH_H__

#include "autodifferentation.h"

// ADNODE GRAPH IMPLEMENTATION
// Graph Structure
typedef struct ComputeGraph{
    struct ComputeGraph* self;
    ADNode** nodes;
    size_t num_nodes;
    size_t capacity;
    void (*add_node)(struct ComputeGraph* self, ADNode* node);
    void (*free)(struct ComputeGraph* self);
    void (*sort)(struct ComputeGraph* self);
    void (*backward)(struct ComputeGraph* self);
}ComputeGraph;  


// Graph Operations
void add_node_to_graph(ComputeGraph* self, ADNode* node){
    if (self->num_nodes == self->capacity){
        self->capacity *= 2;
        self->nodes = (ADNode**)realloc(self->nodes, self->capacity * sizeof(ADNode*));
    }
    self->nodes[self->num_nodes++] = node;
};

void free_graph(ComputeGraph* self){
    if (self){
        for (size_t i=0; i < self->num_nodes; i++){
            ADNode* node = self->nodes[i];
            node->destroy(node);

        }
        
        free(self->nodes);
        free(self);
    }
};

static void dfs(ADNode* node, ADNode** sorted, size_t* idx){
    if (node->visited) return;
    node->visited = 1;

    for (size_t i = 0; i < node->num_parents; i++){
        dfs(node->parents[i], sorted, idx);
    }

    node->topology_idx = (*idx)--; // going backward means indices of nodes are lower
    sorted[node->topology_idx] = node;
};

void topological_sort(ComputeGraph* graph){
    ADNode** sorted = (ADNode**)malloc(graph->num_nodes * sizeof(ADNode*));
    size_t idx = graph->num_nodes - 1;

    // Set all nodes as unvisited
    for (size_t i = 0; i < graph->num_nodes; i++){
        graph->nodes[i]->visited = 0;
    }

    // Traverse and add nodes to corresponding places in sorted array
    for (size_t i = 0; graph->num_nodes; i++){
        if (!graph->nodes[i]->visited) dfs(graph->nodes[i], sorted, &idx);
    }

    // Replace the original array with the sorted one
    free(graph->nodes);
    graph->nodes = sorted;
};

void backward_pass(ComputeGraph* self){
    //perform topological sort
    topological_sort(self);

    // Set gradient of the output Node to 1
    self->nodes[self->num_nodes - 1]->data.grad = 1.0;

    // Perform backward pass
    for (size_t i = self->num_nodes - 1; i >= 0; i--){
        ADNode* node = self->nodes[i];
        if (node->backward) node->backward(node);
    }
};

ComputeGraph* graph_new(){
    ComputeGraph* graph = (ComputeGraph*)malloc(sizeof(ComputeGraph));
    graph->capacity = 10; // start with space for 10 Nodes
    graph->nodes = (ADNode**)malloc(graph->capacity * sizeof(ComputeGraph*));
    graph->num_nodes = 0;

    // Set methods
    graph->add_node = add_node_to_graph;
    graph->free = free_graph;
    graph->sort = topological_sort;
    graph->backward = backward_pass;
    return graph; 
};
#pragma endregion Computation Graph

#endif //  __COMPUTE_GRAPH_H__