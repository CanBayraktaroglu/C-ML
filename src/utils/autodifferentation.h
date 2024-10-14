#ifndef __AUTODIFF_H__
#define __AUTODIFF_H__

#include <stdlib.h>
#include <math.h>
#include "stdio.h"
#include "string.h"

// Core data structure for storing value and derivative
typedef struct {
    double value;
    double grad;
} Dual;

// Node in the computational graph
typedef struct ADNode {
    struct ADNode* self;    
    struct ADNode** parents;
    Dual data;
    size_t num_parents;
    size_t topology_idx;
    char visited;
    char is_trainable;

    // Methods
        void (*backward)(struct ADNode* self);
        void (*destroy)(struct ADNode* self);
        void (*set_val)(struct ADNode* self, const double val);
        void (*set_grad)(struct ADNode* self, const double grad);
        void (*set_parent)(struct ADNode* self, struct ADNode* parent, const size_t parent_idx);
        void (*init)(struct ADNode* self);
    
        double (*get_val)(struct ADNode* self);
        double (*get_grad)(struct ADNode* self);
        
        struct ADNode* (*copy)(struct ADNode* self);
        
        struct ADNode* (*add)(struct ADNode* self, struct ADNode* node);
        struct ADNode* (*multiply)(struct ADNode* self, struct ADNode* node);
        struct ADNode* (*subtract)(struct ADNode* self, struct ADNode* node);
        struct ADNode* (*sqrt)(struct ADNode* self);
        struct ADNode* (*exp)(struct ADNode* self);
        struct ADNode* (*log)(struct ADNode* self);
        struct ADNode* (*sigmoid)(struct ADNode* self); 
        struct ADNode* (*tanh)(struct ADNode* self);


}ADNode;

void node_init(ADNode* self);

ADNode* node_new(const double value, const size_t num_parents, char is_trainable){
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
    node->is_trainable = is_trainable;
    node->backward = NULL;
    node->visited = 0;
    node->init = node_init;
    node->init(node);

    return node;
};

ADNode* node_copy(ADNode* self){
    ADNode* node = (ADNode*)malloc(sizeof(ADNode));
    if (node == NULL){
        printf("Failed to allocate memory for AD Node.\n");
        exit(1);
    }
    
    memcpy(node, self, sizeof(ADNode));
    return node;
};

void node_destroy(ADNode* self){
    if (self){
        if (self->parents){
            free(self->parents);
            self->parents = NULL;
        }

        free(self);
        self = NULL;
    }
};

// Basic backward operations
void backward_add(ADNode* node){
    for (size_t i = 0; i < node->num_parents; i++){
        node->parents[i]->data.grad += node->data.grad;
    }
};

void backward_sqrt(ADNode* node){
    const double da_dzi = 0.5 * (1.0 / sqrt(node->get_grad(node)));
    node->parents[0]->data.grad += node->get_grad(node) * da_dzi; 
};

void backward_multiply(ADNode* node){
    for (size_t i = 0; i < node->num_parents; i++){
        const double da_dzi = node->get_val(node) / node->parents[i]->data.value; 
        node->parents[i]->data.grad += node->get_grad(node) * da_dzi;
    }

};

void backward_exp(ADNode* node){
    const double da_dzi = exp(node->get_val(node));
    node->parents[0]->data.grad += da_dzi;
};

void backward_log(ADNode* node){
    const double da_dzi = 1.0 / node->get_val(node);
    node->parents[0]->data.grad += node->get_grad(node) * da_dzi;
};

void backward_subtract(ADNode* node){
    node->parents[0]->data.grad += node->get_grad(node);
    node->parents[1]->data.grad -= node->get_grad(node);  
};

void backward_sigmoid(ADNode* node){
    const double da_dzi = node->get_val(node) * (1 - node->get_val(node));
    node->parents[0]->data.grad += node->get_grad(node) * da_dzi;   
};

void backward_tanh(ADNode* self){
    const double da_dzi = 1 - pow(tanh(self->get_val(self)), 2);
    self->parents[0]->data.grad += self->get_grad(self) * da_dzi;
}

// ... (other operations will be added here)

// Function prototypes

void node_set_parent(ADNode* self, ADNode* parent, const size_t parent_idx){
    if (self == NULL){
        printf("Node*::self points to null in method node_set_parent.\n");
        return;
    }

    if (parent == NULL){
        printf("Node*::parent points to null in method node_set_parent.\n");
        return;
    }

    if (parent_idx >= self->num_parents){
        printf("provided idx %lu exceeds %lu, the number of parents in node_set_parent.", parent_idx, self->num_parents);
        return;
    }

    self->parents[parent_idx] = parent;
};

void node_set_val(ADNode* self, const double val){
    self->data.value = val;
};

void node_set_grad(ADNode* self, const double grad){
    self->data.grad = grad;
};

double node_get_val(ADNode* self){
    return self->data.value;
};

static double node_get_grad(ADNode* self){
    return self->data.grad;
}

ADNode* node_add(ADNode* self, ADNode* node){

    ADNode* result = node_new(self->get_val(self) + node->get_val(node), 2, 0);

    result->parents[0] = self;
    result->parents[1] = node;
    result->backward = backward_add;
    return result;
};

ADNode* node_multiply(ADNode* self, ADNode* node){
    ADNode* result = node_new(self->get_val(self) * node->get_val(node), 2, 0);

    result->parents[0] = self;
    result->parents[1] = node;
    result->backward = backward_multiply;
    return result;
};

ADNode* node_sqrt(ADNode* self){
    ADNode* result = node_new(sqrt(self->get_val(self)), 1, 0); 
    result->parents[0] = self;
    result->backward = backward_sqrt;
    return result;
};

ADNode* node_exp(ADNode* self){
    ADNode* result = node_new(exp(self->get_val(self)), 1, 0); 
    result->parents[0] = self;
    result->backward = backward_exp;
    return result;
};

ADNode* node_log(ADNode* self){
    ADNode* result = node_new(log(self->get_val(self)), 1, 0); 
    result->parents[0] = self;
    result->backward = backward_log;
    return result;
};

ADNode* node_sigmoid(ADNode* self){
    ADNode* result = node_new(1/(1 + exp(-self->get_val(self))), 1, 0);
    result->parents[0] = self;
    result->backward = backward_sigmoid;
    return result;
};

ADNode* node_tanh(ADNode* self){
    ADNode* result = node_new(tanh(self->get_val(self)), 1, 0);
    result->parents[0] = self;
    result->backward = backward_tanh;
    return result;
};

ADNode* node_subtract(ADNode* self, ADNode* node){
    ADNode* result = node_new(self->get_val(self) - node->get_val(node), 2, 0);
    result->parents[0] = self;
    result->parents[1] = node;
    result->backward = backward_subtract;
    return result;
};

void node_init(ADNode* self){
    self->destroy = node_destroy;
    self->set_val = node_set_val;
    self->set_grad = node_set_grad;
    self->set_parent = node_set_parent;
    self->get_val = node_get_val;
    self->get_grad = node_get_grad;
    self->add = node_add;
    self->multiply = node_multiply;
    self->subtract = node_subtract;
    self->sqrt = node_sqrt;
    self->exp = node_exp;
    self->log = node_log;
    self->copy = node_copy;
    self->sigmoid = node_sigmoid;
    self->tanh = node_tanh;
};

#pragma region Computation Graph

#endif // AUTODIFF_H