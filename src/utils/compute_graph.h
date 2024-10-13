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
    Adam_Optimizer* optimizer;

    void (*add_node)(struct ComputeGraph* self, ADNode* node);
    void (*free)(struct ComputeGraph* self);
    void (*sort)(struct ComputeGraph* self);
    void (*propagate_back)(struct ComputeGraph* self);
    void (*prune)(struct ComputeGraph* self);
    void (*optimize)(struct ComputeGraph* self);

}ComputeGraph;  


// Graph Operations
void add_node_to_graph(ComputeGraph* self, ADNode* node){
    if (self->num_nodes == self->capacity){
        self->capacity *= 2;
        self->nodes = (ADNode**)realloc(self->nodes, self->capacity * sizeof(ADNode*));
    }
    self->nodes[self->num_nodes++] = node;
};

void graph_prune(ComputeGraph* self){
    if (self){
        for (size_t i=0; i < self->num_nodes; i++){
            ADNode* node = self->nodes[i];
            if (node->is_trainable) continue;

            node->destroy(node);

        }
        
        free(self->nodes);
    }
};

void graph_free(ComputeGraph* self){
    if (self){
        for (size_t i=0; i < self->num_nodes; i++){
            ADNode* node = self->nodes[i];
            node->destroy(node);

        }
        
        free(self->nodes);
        free(self);
    }
};

void dfs_sort(ADNode* node, ADNode** sorted, size_t* idx){
    if (node == NULL || node->visited) return;
    node->visited = 1;

    for (size_t i = 0; i < node->num_parents; i++){
        dfs_sort(node->parents[i], sorted, idx);
    }

    node->topology_idx = (*idx)--; 
    sorted[node->topology_idx] = node;
};

void dfs_explore(ComputeGraph* graph, ADNode* node){
    if (node == NULL || node->visited) return;
    node->visited = 1;
    graph->add_node(graph, node);

    for (size_t i = 0; i < node->num_parents; i++){
        dfs_explore(graph, node->parents[i]);
    }

};

void dfs_backward(ADNode* node){
    if (node == NULL || node->visited) return;
    node->visited = 1;
    if (node->backward) node->backward(node);

    for (size_t i = 0; i < node->num_parents; i++){
        dfs_backward(node->parents[i]);
    }

};

void graph_topological_sort(ComputeGraph* graph){
    ADNode** sorted = (ADNode**)malloc(graph->num_nodes * sizeof(ADNode*));
    size_t idx = graph->num_nodes - 1;

    // Set all nodes as unvisited
    for (size_t i = 0; i < graph->num_nodes; i++){
        graph->nodes[i]->visited = 0;
    }

    // Traverse and add nodes to corresponding places in sorted array
    for (size_t i = 0; graph->num_nodes; i++){
        if (!graph->nodes[i]->visited) dfs_sort(graph->nodes[i], sorted, &idx);
    }

    // Replace the original array with the sorted one
    free(graph->nodes);
    graph->nodes = sorted;
};

void graph_propagate_back(ComputeGraph* self){

    // Set gradient of the output Node to 1
    self->head->data.grad = 1.0;

    // Set all nodes to unvisited
    for (size_t i = 0; i < self->num_nodes; i++){
        self->nodes[i]->visited = 0;
    }

    // Traverse graph and propagate back
    dfs_backward(self->head);
    
};

void graph_optimize(ComputeGraph* self){
    return;
};

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
    dfs_explore(graph, graph->head);
};      

ComputeGraph* graph_new(){
    ComputeGraph* graph = (ComputeGraph*)malloc(sizeof(ComputeGraph));
    graph->capacity = 10; // start with space for 10 Nodes
    graph->nodes = (ADNode**)malloc(graph->capacity * sizeof(ComputeGraph*));
    graph->num_nodes = 0;

    // Set methods
    graph->add_node = add_node_to_graph;
    graph->free = graph_free;
    graph->sort = graph_topological_sort;
    graph->propagate_back = graph_propagate_back;
    graph->prune = graph_prune;
    graph->optimize = graph_optimize;

    return graph; 
};
#pragma endregion Computation Graph

#endif //  __COMPUTE_GRAPH_H__