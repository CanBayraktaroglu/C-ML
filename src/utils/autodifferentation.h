#ifndef __AUTODIFF_H__
#define __AUTODIFF_H__

#include <stdlib.h>
#include <math.h>

// Core data structure for storing value and derivative
typedef struct {
    double value;
    double grad;
} Dual;

// Node in the computational graph
typedef struct ADNode {
    ADNode* self;    
    ADNode** parents;
    Dual data;
    int num_parents;
    size_t topology_idx;
    char visited;
    void (*backward)(ADNode* self);
    void (*free)(ADNode* self);
    void (*set_val)(ADNode* self, const double val);
    void (*set_grad)(ADNode* self, const double grad);
    double (*get_val)(ADNode* self);
    double (*get_grad)(ADNode* self);

}ADNode;

void free_node(ADNode* node){
    if (node){
        free(node->parents);
        free(node);
    }
};

// Basic backward operations
static void backward_add(ADNode* node){
    node->parents[0]->data.grad += node->data.grad;
    node->parents[1]->data.grad += node->data.grad;
};


static void backward_multiply(ADNode* node){
    node->parents[0]->data.grad += node->data.grad * node->parents[1]->data.value;
    node->parents[1]->data.grad += node->data.grad * node->parents[0]->data.value;
};


static void backward_subtract(ADNode* node){
    node->parents[0]->data.grad += node->data.grad;
    node->parents[1]->data.grad -= node->data.grad;  
};


// ... (other operations will be added here)

// Function prototypes
static void node_set_val(ADNode* self, const double val){
    self->data.value = val;
};

static void node_set_grad(ADNode* self, const double grad){
    self->data.grad = grad;
};

static double node_get_val(ADNode* self){
    return self->data.value;
};

static double node_get_grad(ADNode* self){
    return self->data.grad;
}

ADNode* node_new(double value, int num_parents) {
    ADNode* node = (ADNode*)malloc(sizeof(ADNode));
    node->data.value = value;
    node->data.grad = 0.0;
    node->num_parents = num_parents;
    if (num_parents > 0) {
        node->parents = (ADNode**)malloc(num_parents * sizeof(ADNode*));
    } else {
        node->parents = NULL;
    }
    
    //Set Methods
    node->backward = NULL;
    node->free = free_node;
    node->set_val = node_set_val;
    node->set_grad = node_set_grad;
    node->get_val = node_get_val;
    node->get_grad = node_get_grad;

    return node;
}


ADNode* add(ADNode* a, ADNode* b){
    ADNode* result = node_new(a->data.value + b->data.value, 2);
    result->parents[0] = a;
    result->parents[1] = b;
    result->backward = backward_add;
    return result;
}
ADNode* multiply(ADNode* a, ADNode* b){
    ADNode* result = node_new(a->data.value * b->data.value, 2);
    result->parents[0] = a;
    result->parents[1] = b;
    result->backward = backward_multiply; 
};
ADNode* subtract(ADNode* a, ADNode* b){
    ADNode* result = node_new(a->data.value - b->data.value, 2);
    result->parents[0] = a;
    result->parents[1] = b;
    result->backward = backward_subtract;
};

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
            const ADNode* node = self->nodes[i];
            node->free(node);

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
#endif // AUTODIFF_H