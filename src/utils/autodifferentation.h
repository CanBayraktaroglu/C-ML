#ifndef AUTODIFF_H
#define AUTODIFF_H

#include <stdlib.h>
#include <math.h>

// Forward declarations
struct ADNode;
typedef struct ADNode ADNode;

// Core data structure for storing value and derivative
typedef struct {
    double value;
    double grad;
} Dual;

// Node in the computational graph
struct ADNode {
    struct ADNode* self;    
    Dual data;
    ADNode** parents;
    int num_parents;
    void (*backward)(ADNode* self);
    size_t topology_idx;
    char visited;
};

ADNode* create_node(double value, int num_parents) {
    ADNode* node = (ADNode*)malloc(sizeof(ADNode));
    node->data.value = value;
    node->data.grad = 0.0;
    node->num_parents = num_parents;
    if (num_parents > 0) {
        node->parents = (ADNode**)malloc(num_parents * sizeof(ADNode*));
    } else {
        node->parents = NULL;
    }
    node->backward = NULL;
    return node;
}

// Function prototypes
ADNode* create_variable(double value){
    return create_node(value, 0);
};

ADNode* create_constant(double value){
    ADNode* node = create_node(value, 0);

    // Constants -> no further autograd
    node->backward = NULL;
    return node;
};

void free_node(ADNode* node){
if (node){
    free(node->parents);
    free(node);
}
};

// Basic operations
static void backward_add(ADNode* node){
    node->parents[0]->data.grad += node->data.grad;
    node->parents[1]->data.grad += node->data.grad;
};

ADNode* add(ADNode* a, ADNode* b){
    ADNode* result = create_node(a->data.value + b->data.value, 2);
    result->parents[0] = a;
    result->parents[1] = b;
    result->backward = backward_add;
    return result;
}

static void backward_multiply(ADNode* node){
    node->parents[0]->data.grad += node->data.grad * node->parents[1]->data.value;
    node->parents[1]->data.grad += node->data.grad * node->parents[0]->data.value;
};

ADNode* multiply(ADNode* a, ADNode* b){
    ADNode* result = create_node(a->data.value * b->data.value, 2);
    result->parents[0] = a;
    result->parents[1] = b;
    result->backward = backward_multiply; 
};

static void backward_subtract(ADNode* node){
    node->parents[0]->data.grad += node->data.grad;
    node->parents[1]->data.grad -= node->data.grad;  
};

ADNode* subtract(ADNode* a, ADNode* b){
    ADNode* result = create_node(a->data.value - b->data.value, 2);
    result->parents[0] = a;
    result->parents[1] = b;
    result->backward = backward_subtract;
};

// ... (other operations will be added here)

// ADNODE GRAPH IMPLEMENTATION
// Graph Structure
typedef struct ComputeGraph{
    struct ComputeGraph* self;
    ADNode** nodes;
    size_t num_nodes;
    size_t capacity;
    void (*add_node_to_graph)(ComputeGraph* self, ADNode* node);
    void (*free_graph)(ComputeGraph* self);
    void (*topological_sort)(ComputeGraph* self);
    void (*backward)(ComputeGraph* self);
}ComputeGraph;  

// Graph Operations
ComputeGraph* graph_new(){
    ComputeGraph* graph = (ComputeGraph*)malloc(sizeof(ComputeGraph));
    graph->capacity = 10; // start with space for 10 Nodes
    graph->nodes = (ADNode**)malloc(graph->capacity * sizeof(ComputeGraph*));
    graph->num_nodes = 0;

    // Set methods
    graph->add_node_to_graph = add_node_to_graph;
    graph->free_graph = free_graph;
    graph->topological_sort = topological_sort;
    graph->backward = backward;
    return graph; 
};

void add_node_to_graph(ComputeGraph* graph, ADNode* node){
    if (graph->num_nodes == graph->capacity){
        graph->capacity *= 2;
        graph->nodes = (ADNode**)realloc(graph->nodes, graph->capacity * sizeof(ADNode*));
    }
    graph->nodes[graph->num_nodes++] = node;
};

void free_graph(ComputeGraph* graph){
    for (size_t i=0; i < graph->num_nodes; i++){
        free_node(graph->nodes[i]);
    }
    free(graph->nodes);
    free(graph);
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

void backward(ComputeGraph* graph){
    //perform topological sort
    topological_sort(graph);

    // Set gradient of the output Node to 1
    graph->nodes[graph->num_nodes - 1]->data.grad = 1.0;

    // Perform backward pass
    for (size_t i = graph->num_nodes - 1; i >= 0; i--){
        ADNode* node = graph->nodes[i];
        if (node->backward) node->backward(node);
    }
};

#endif // AUTODIFF_H