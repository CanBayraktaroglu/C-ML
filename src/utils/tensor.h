#ifndef __TENSOR_H__
#define __TENSOR_H__

#include <stdlib.h>
#include <string.h>
#include "point.h"
#include "autodifferentation.h"


typedef struct Tensor{
    
    // Attributes
        struct Tensor* self;
        ADNode** nodes;
        size_t n_rows;
        size_t n_cols;
    
    // Methods
        void (*realloc)(struct Tensor* self, const size_t n_rows, const size_t n_cols);
        void (*init)(struct Tensor* self);
        void (*destroy)(struct Tensor* self);
        void (*detach)(struct Tensor* self);
        void (*print_val)(struct Tensor* self);
        void (*print_grad)(struct Tensor* self);

        void (*transpose_inplace)(struct Tensor* self); 
        void (*abs_inplace)(struct Tensor* self);
        void (*sqrt_inplace)(struct Tensor* self);
        void (*exp_inplace)(struct Tensor* self);
        void (*log_inplace)(struct Tensor* self);

        struct Tensor* (*transpose)(struct Tensor* self); 
        struct Tensor* (*copy)(struct Tensor* self);
        struct Tensor* (*abs)(struct Tensor* self);
        struct Tensor* (*sqrt)(struct Tensor* self);
        struct Tensor* (*exp)(struct Tensor* self);
        struct Tensor* (*log)(struct Tensor* self);
        struct Tensor* (*relu)(struct Tensor* self);
        struct Tensor* (*sigmoid)(struct Tensor* self);
        struct Tensor* (*tanh)(struct Tensor* self);

        double (*froebenius_norm)(struct Tensor* self);

        // Getters
        double (*get_val)(struct Tensor* self, const size_t i, const size_t j);
        double (*get_grad)(struct Tensor* self, const size_t i, const size_t j);
        ADNode* (*get_node)(struct Tensor* self, const size_t i, const size_t j);
        
        // Setters
        void (*set_val)(struct Tensor* self, const size_t i, const size_t j, const double val);
        void (*set_grad)(struct Tensor* self, const size_t i, const size_t j, const double grad);
        void (*set_node)(struct Tensor* self, ADNode* node, const size_t i, const size_t j);
        
} Tensor;

void tensor_init(Tensor* self);

Tensor* tensor_new(const size_t n_rows, const size_t n_cols){
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    
    if (tensor == NULL){
        printf("Failed to allocate memory for Tensor.\n");
        exit(1);
    }

    tensor->self = tensor;
    tensor->nodes = (ADNode**)malloc(n_rows * n_cols * sizeof(ADNode*));
    tensor->n_rows = n_rows;
    tensor->n_cols = n_cols;

    tensor->init = tensor_init;
    tensor->init(tensor);

    for(size_t i = 0; i < n_rows; i++){
        for(size_t j = 0; j < n_cols; j++){
            tensor->nodes[i * n_cols + j] = node_new(0.0, 0, 0);
        }
    };

    return tensor;
};

void tensor_realloc(Tensor* self, const size_t n_rows, const size_t n_cols){
    self->nodes = (ADNode**) realloc(self->nodes, n_rows * n_cols * sizeof(ADNode*));
    
    /* for (size_t i = 0; i < n_rows; i++){
        for (size_t j = 0; j < n_cols; j++){
            ADNode* node = self->nodes[i * n_cols + j];
            if (node == NULL){
                self->nodes[i * n_cols + j] = node_new(0.0, 0, 0);
            }
        }
    } */ 

    self->n_rows = n_rows;
    self->n_cols = n_cols;
};

static ADNode* tensor_get_node(Tensor* self, const size_t i, const size_t j){
    return self->nodes[i *  self->n_cols + j];
};

void tensor_set_node(Tensor* self, ADNode* node, const size_t i, const size_t j){
    self->nodes[i * self->n_cols + j] = node;
};

void tensor_destroy(Tensor* self){
    if (self){
        for (size_t i = 0; i < self->n_rows; i++){
            for (size_t j = 0; j < self->n_cols; j++){
                ADNode* node = self->get_node(self, i, j);
                node->destroy(node);
                node = NULL;
                self->set_node(self, node, i, j);

            }
        }
        free(self->nodes);
        free(self);
    }

};

void tensor_detach(Tensor* self){
    if (self){
        free(self->nodes);
        self->nodes = NULL;
        free(self);
     }
};

void tensor_set_val(Tensor* self, const size_t i, const size_t j, const double val){
    if (i >= self->n_rows){
        printf("row index exceeded tensor row number.\n");
        exit(0);
    }

    if (j >= self->n_cols){
        printf("col index exceeded tensor col number.\n");
        exit(0);
    }
    ADNode* node = self->nodes[i * self->n_cols + j];
    node->set_val(node, val);

};

void tensor_set_grad(Tensor* self, const size_t i, const size_t j, const double grad){
    if (i >= self->n_rows){
        printf("row index exceeded tensor row number.\n");
        exit(0);
    }
    
    if (j >= self->n_cols){
        printf("col index exceeded tensor col number.\n");
        exit(0);
    }
    ADNode* node = self->nodes[i * self->n_cols + j];
    node->set_grad(node, grad);
};

static double tensor_get_val(Tensor* self, const size_t i, const size_t j){
    if (i >= self->n_rows){
        printf("row index exceeded tensor row number.\n");
        exit(0);
    }
    
    if (j >= self->n_cols){
        printf("col index exceeded tensor col number.\n");
        exit(0);
    }

    ADNode* node = self->nodes[i * self->n_cols + j];
    return node->get_val(node);

};

static double tensor_get_grad(Tensor* self, const size_t i, const size_t j){
    if (i >= self->n_rows){
        printf("row index exceeded tensor row number.\n");
        exit(0);
    }
    
    if (j >= self->n_cols){
        printf("col index exceeded tensor col number.\n");
        exit(0);
    }

    ADNode* node = self->nodes[i * self->n_cols + j];
    return node->get_grad(node);

};

// Nodes are Shared
void tensor_transpose_inplace(Tensor* self){
    ADNode* temp = (ADNode*)malloc(sizeof(ADNode));
    
    for(size_t i = 0; i < self->n_rows; i++){
        for(size_t j = 0; j < self->n_cols; j++){
            ADNode* source = self->nodes[i * self->n_cols + j]; // iter cols
            ADNode* target = self->nodes[j * self->n_rows + i];

            memcpy(temp, source, sizeof(ADNode));
            memcpy(source, target, sizeof(ADNode));
            memcpy(target, temp, sizeof(ADNode));
        }
    }
    
    temp->destroy(temp);        
};

Tensor* tensor_transpose(Tensor* self){
    Tensor* temp = tensor_new(self->n_cols, self->n_rows);

    for(size_t i = 0; i < self->n_rows; i++){
        for(size_t j = 0; j < self->n_cols; j++){
            ADNode* source = self->get_node(self, i, j); // iter cols
            temp->set_node(temp, source, j,i);
        }
    }

    return temp;
};

Tensor* tensor_scalar_product(Tensor* self, const double scalar){
   Tensor* result = tensor_new(self->n_rows, self->n_cols); 

    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* source_node = self->get_node(self, i, j);
            ADNode* constant_node = node_new(scalar, 0, 0);
            ADNode* resulting_node = node_multiply(source_node, constant_node);
            result->set_node(result, resulting_node, i, j);
        }
    }
    return result;
};

void tensor_scalar_product_inplace(Tensor* self, const double scalar){ 
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
                        
            ADNode* source_node = self->get_node(self, i, j);
            ADNode* constant_node = node_new(scalar, 0, 0);
            ADNode* target_node = node_multiply(source_node, constant_node);
            self->set_node(self, target_node, i, j);
        }
    }
};

Tensor* tensor_add(Tensor* self, Tensor* tensor){
    if (self == NULL){
        printf("Tensor a is pointing to an empty address\n.");
        exit(0);
    }

    if (tensor == NULL){
        printf("Tensor b is pointing to an empty address\n.");
        exit(0);
    }

    if (self->n_rows != tensor->n_rows || self->n_cols != tensor->n_cols){
        printf("Tensor dimensions do not match for addition\n.");
        exit(0);
    }

    Tensor* result = tensor_new(self->n_rows, self->n_cols);

    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node_A = self->get_node(self, i, j);
            ADNode* node_B = tensor->get_node(tensor, i, j);
            ADNode* resulting_node = node_add(node_A, node_B);

            result->set_node(result, resulting_node, i, j);
        }
    }

    return result;
};

void tensor_add_inplace(Tensor* self, Tensor* tensor){

    if (self->n_rows != tensor->n_rows || self->n_cols != tensor->n_cols){
        printf("Tensor dimensions do not match for addition.\n");
        printf("self->n_rows: %lu, tensor->n_rows: %lu\n", self->n_rows, tensor->n_rows);
        printf("self->n_cols: %lu, tensor->n_cols: %lu\n", self->n_cols, tensor->n_cols);
        exit(0);
    }
    
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node_A = self->get_node(self, i, j);
            ADNode* node_B = tensor->get_node(tensor, i, j);
            ADNode* resulting_node = node_add(node_A, node_B);
            self->set_node(self, resulting_node, i, j);
        }
    }
};

Tensor* tensor_subtract(Tensor* self, Tensor* tensor){
    if (self == NULL){
        printf("Tensor a is pointing to an empty address\n.");
        exit(0);
    }

    if (tensor == NULL){
        printf("Tensor b is pointing to an empty address\n.");
        exit(0);
    }

    if (self->n_rows != tensor->n_rows || self->n_cols != tensor->n_cols){
        printf("Tensor dimensions do not match for subtraction.\n");
        exit(0);
    }

    Tensor* result = tensor_new(self->n_rows, self->n_cols);
    
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node_A = self->get_node(self, i, j);
            ADNode* node_B = tensor->get_node(tensor, i, j);
            
            ADNode* resulting_node = node_subtract(node_A, node_B);
            result->set_node(result, resulting_node, i, j);
        }
    }

    return result;
};

void tensor_subtract_inplace(Tensor* self, Tensor* tensor){
    if (self == NULL){
        printf("Tensor a is pointing to an empty address\n.");
        return;
    }

    if (tensor == NULL){
        printf("Tensor b is pointing to an empty address\n.");
        return;
    }

    if (self->n_rows != tensor->n_rows || self->n_cols != tensor->n_cols){
        printf("Tensor dimensions do not match for subtraction.\n");
        exit(0);
    }
    
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node_A = self->get_node(self, i, j);
            ADNode* node_B = tensor->get_node(tensor, i, j);
            
            ADNode* resulting_node = node_subtract(node_A, node_B);
            self->set_node(self, resulting_node, i, j);
        }
    }

};

void tensor_print_val(Tensor* self){
    for(size_t i = 0; i < self->n_rows; i++){
        for(size_t j = 0; j < self->n_cols; j++){
            ADNode* node = self->get_node(self, i, j);
            if (node == NULL){
                printf("NULL ");
            }
            const double val = self->get_val(self, i, j);
            printf("%f ", val);
        }
        printf("\n");
    }
};

void tensor_print_grad(Tensor* self){
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            const double grad = self->get_grad(self, i, j);
            printf("%f ", grad);  
        }
        printf("\n");
    }
}

Tensor* tensor_dot_product(Tensor* self, Tensor* tensor){
    if (self == NULL){
        printf("Tensor a is pointing to an empty address\n.");
        exit(0);
    }

    if (tensor == NULL){
        printf("Tensor b is pointing to an empty address\n.");
        exit(0);
    }

    if (self->n_cols != tensor->n_rows){
        printf("Tensor dimensions do not match for multiplication.\n");
        exit(0);
    }
    
    Tensor* result = tensor_new(self->n_rows, tensor->n_cols); 
    
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < tensor->n_cols; j++){
            
            ADNode* result_node = node_new(0.0, self->n_cols, 0);

            for (size_t k = 0; k < self->n_cols; k++){
                ADNode* self_node = self->get_node(self, i, k);
                ADNode* tensor_node = tensor->get_node(tensor, k, j);
                ADNode* product_node = node_multiply(self_node, tensor_node); 
                result_node->set_parent(result_node, product_node, k);
                result_node->data.value += product_node->get_val(product_node);
            }

            // Set backward
            result_node->backward = backward_add;

            // set the resulting node
            result->set_node(result, result_node, i, j);         
            
        }
    }
    return result;  
};

void tensor_dot_product_inplace(Tensor* self, Tensor* tensor){
    if (self == NULL){
        printf("Tensor a is pointing to an empty address\n.");
        return;
    }

    if (tensor == NULL){
        printf("Tensor b is pointing to an empty address\n.");
        return;
    }

    if ((self)->n_cols != tensor->n_rows){
        printf("Tensor dimensions do not match for multiplication.\n");
        exit(0);
    }

    Tensor* tmp = (self)->copy(self);
    
    // reallocate_memory
    self->realloc(self, self->n_rows, tensor->n_cols);

    for (size_t i = 0; i < tmp->n_rows; i++){
        for (size_t j = 0; j < tensor->n_cols; j++){
            ADNode* result_node = node_new(0.0, self->n_cols, 0);

            for (size_t k = 0; k < tmp->n_cols; k++){
                ADNode* self_node = tmp->get_node(tmp, i, k);
                ADNode* tensor_node = tensor->get_node(tensor, k, j);
                ADNode* product_node = node_multiply(self_node, tensor_node);
                result_node->set_parent(result_node, product_node, k);
                result_node->data.value += product_node->get_val(result_node);
            }

            // Set backward
            result_node->backward = backward_add;

            // set the resulting node
            self->set_node(self, result_node, i, j); 
        }
    }
    
};

void tensor_dot_product_reversed_order_inplace(Tensor* self, Tensor* tensor){
    
    if (self == NULL){
        printf("Tensor a is pointing to an empty address\n.");
        return;
    }

    if (tensor == NULL){
        printf("Tensor b is pointing to an empty address\n.");
        return;
    }


    if (tensor->n_cols != self->n_rows){
        printf("Tensor dimensions do not match for dot product.\n");
        exit(0);
    }

    // Copy self
    Tensor* tmp = self->copy(self);

    // Free nodes in self
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node = self->get_node(self, i, j);
            node->destroy(node);
            node = NULL;
            self->set_node(self, node, i, j);
        }
    }

    // reallocate_memory
    // tensor n_rows, self n_cols
    if (self->n_rows != tensor->n_rows){
        self->realloc(self, tensor->n_rows, self->n_cols);
    }

    for (size_t i = 0; i < tensor->n_rows; i++){
        for (size_t j = 0; j < tmp->n_cols; j++){
            ADNode* result_node = node_new(0.0, tensor->n_cols, 0);

            for (size_t k = 0; k < tensor->n_cols; k++){
                ADNode* tensor_node = tensor->get_node(tensor, i, k);
                ADNode* self_node = tmp->get_node(tmp, k, j);
                ADNode* product_node = node_multiply(tensor_node, self_node);

                result_node->set_parent(result_node, product_node, k);
                result_node->data.value += product_node->get_val(result_node);
            }

            // Set backward
            result_node->backward = backward_add;

            // set the resulting node
            self->set_node(self, result_node, i, j); 

        }
    }

    tmp->detach(tmp);
};

Tensor* tensor_copy(Tensor* self){

    Tensor* tensor = tensor_new(self->n_rows, self->n_cols);
    
    if (tensor == NULL){
        printf("Failed to allocate memory for Tensor.\n");
        exit(1);
    }

    
    if (tensor->nodes == NULL){
        printf("Failed to allocate memory for Tensor nodes.\n");
        free(tensor);
        exit(1);
    }

    ADNode* node = NULL;
    ADNode* new_node = NULL;
    ADNode* tensor_node = NULL;
    
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            node = self->get_node(self, i, j);
            new_node = node->copy(node);

            // TODO FREE ONLY INITIALIZED NODES
            tensor_node = tensor->get_node(tensor, i, j);
            tensor_node->destroy(tensor_node);
            
            // Set the new node
            tensor->set_node(tensor, new_node, i, j);
        }
    }

    return tensor;
};

void tensor_abs_inplace(Tensor* self){
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node = self->get_node(self, i, j);
            if (node->get_val(node) < 0){
                ADNode* constant_node = node_new(-1.0, 0, 0);
                ADNode* result_node = node_multiply(node, constant_node);
                self->set_node(self, result_node, i, j);
            }
        }
    }
}; 

Tensor* tensor_abs(Tensor* self){
    Tensor* tensor = tensor_new(self->n_rows, self->n_cols);
    printf("Copying tensor.\n");

    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node = self->get_node(self, i, j);
            ADNode* result_node = NULL;
            ADNode* constant_node = NULL; 

            if (node->get_val(node) < 0){
                constant_node = node_new(-1.0, 0, 0);
            }
            else {
                constant_node = node_new(1.0, 0, 0);
            }
            result_node = node_multiply(node, constant_node);
            tensor->set_node(tensor, result_node, i, j);
        }
    }

    return tensor;
};

Tensor* tensor_relu(Tensor* self){
    Tensor* result = tensor_new(self->n_rows, self->n_cols);
    ADNode* n = NULL;

    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node = self->get_node(self, i, j);
            if (node->get_val(node) >= 0){
                n = node_new(node->get_val(node), 1, 0);
            } 
            else {
                n = node_new(0.0, 1, 0);
            }
                result->set_node(result, n, i, j);
                n->set_parent(n, node, 0);
        }
    }
    return result;    
};

void tensor_relu_inplace(Tensor* self){
    ADNode* n = NULL;

    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node = self->get_node(self, i, j);
            if (node->get_val(node) >= 0){
                n = node;
            } 
            else {
                n = node_multiply(node, node_new(0.0, 0, 0));
            }

            self->set_node(self, n, i, j);
        }
    }
};

Tensor* tensor_sigmoid(Tensor* self){
    Tensor* result = tensor_new(self->n_rows, self->n_cols);
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node = self->get_node(self, i, j);
            ADNode* result_node = node_sigmoid(node);
            result->set_node(result, result_node, i, j);
        }
    }
    return result;
};

Tensor* tensor_tanh(Tensor* self){
    Tensor* result = tensor_new(self->n_rows, self->n_cols);
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node = self->get_node(self, i, j);
            ADNode* result_node = node_tanh(node);
            result->set_node(result, result_node, i, j);
        }
    }
    return result;
};

Tensor* tensor_create_identity(const size_t n){
    Tensor* identity = tensor_new(n, n);

    for (size_t i = 0; i < n; i++){
        for (size_t j = 0; j < n; j++){
            
            ADNode* node = NULL;
            node = node_new((double)(i==j), 0, 0);
            
            identity->set_node(identity, node, i, j);
        }
    }

    return identity;
}; 

double tensor_froebenius_norm(Tensor* self){
    // euclidian norm of the vector, which is the matrix flattened out
    double sum = 0.0;
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            sum += self->get_val(self, i, j) * self->get_val(self, i, j);
        }
    }
    return sqrt(sum);
}; 

Tensor* tensor_sqrt(Tensor* self){
    Tensor* tensor = tensor_new(self->n_rows, self->n_cols);

    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node = self->get_node(self, i, j);
            ADNode* result_node = node->sqrt(node);
            tensor->set_node(tensor, result_node, i, j);
        }
    }

    return tensor;
};

void tensor_sqrt_inplace(Tensor* self){
    
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node = self->get_node(self, i, j);
            ADNode* result_node = node->sqrt(node);
            self->set_node(self, result_node, i, j);
        }
    }

};

void tensor_exp_inplace(Tensor* self){
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node = self->get_node(self, i, j);
            ADNode* result_node = node->exp(node);
            self->set_node(self, result_node, i, j);
        }
    }
};

Tensor* tensor_exp(Tensor* self){
    Tensor* tensor = tensor_new(self->n_rows, self->n_cols);

    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node = self->get_node(self, i, j);
            ADNode* result_node = node_exp(node);
            tensor->set_node(tensor, result_node, i, j);
        }
    }

    return tensor;
};

void tensor_log_inplace(Tensor* self){
    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node = self->get_node(self, i, j);
            ADNode* result_node = node->log(node);
            self->set_node(self, result_node, i, j);
        }
    }
};

Tensor* tensor_log(Tensor* self){
    Tensor* tensor = tensor_new(self->n_rows, self->n_cols);

    for (size_t i = 0; i < self->n_rows; i++){
        for (size_t j = 0; j < self->n_cols; j++){
            ADNode* node = self->get_node(self, i, j);
            ADNode* result_node = node->log(node);
            tensor->set_node(tensor, result_node, i, j);
        }
    }

    return tensor;
};

Tensor* tensor_create_from_array(const size_t n_rows, const size_t n_cols, const double (*arr)[n_cols]){
    if (arr == NULL){
        printf("Array to be converted to Tensor points to an empty address.\n");
        exit(0);
    }

    Tensor* tensor = tensor_new(n_rows, n_cols);
    
    for (size_t i = 0; i < n_rows; i++){
        for (size_t j = 0; j < n_cols; j++){ 
            // Get and Destroy the tensor node
            ADNode* tensor_node = tensor->get_node(tensor, i, j);
            tensor_node->destroy(tensor_node);
            ADNode* node = node_new(arr[i][j], 0, 0);
            tensor->set_node(tensor, node, i, j);
        }
    }


    return tensor;
}; 


void tensor_init(Tensor* self){
    // Set methods
    
    self->set_val = tensor_set_val;
    self->set_grad = tensor_set_grad;
    self->get_val = tensor_get_val;
    self->get_grad = tensor_get_grad;   
    self->get_node = tensor_get_node;
    self->set_node = tensor_set_node;
    self->print_val = tensor_print_val;
    self->print_grad = tensor_print_grad;
    
    self->realloc = tensor_realloc;
    self->detach = tensor_detach;
    self->destroy = tensor_destroy;
    self->copy = tensor_copy;
    self->transpose = tensor_transpose;
    
    self->abs_inplace = tensor_abs_inplace;
    self->transpose_inplace = tensor_transpose_inplace;
    self->sqrt_inplace = tensor_sqrt_inplace;
    self->exp_inplace = tensor_exp_inplace;
    self->log_inplace = tensor_log_inplace;

};
#endif // __TENSOR_H__