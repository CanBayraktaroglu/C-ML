#ifndef __DT_H__
#define __DT_H__

#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "vector.h"

typedef struct DT_Node{
    size_t feature_idx;
    float threshold;
    struct DT_Node* left;
    struct DT_Node* right;
    float info_gain;
    Vector* vec;
}DT_Node;

DT_Node* dt_node_create(size_t feature_idx, float threshold){
    DT_Node* node = (DT_Node*)malloc(sizeof(DT_Node));
    node->feature_idx = feature_idx;
    node->threshold = threshold;
    node->left = NULL;
    node->right = NULL;
    node->vec = NULL;
    node->info_gain = 0;
    
    return node;
};

void dt_node_set_vec(DT_Node* node, Vector* vec){
    if (node->vec == NULL){
        node->vec = (Vector*)malloc(sizeof(Vector));
    }
    memcpy(node->vec, vec,  sizeof(Vector));
};

void dt_node_destroy(DT_Node* node, size_t depth){
    if (node == NULL) return;
    
    if (!depth){
        for (unsigned short i = 0; i < node->vec->size; i++){
            Point* p = vector_at(node->vec, i);
            point_destroy(&p);
        }
    }

    dt_node_destroy(node->left, depth + 1);
    node->left = NULL;
    dt_node_destroy(node->right, depth + 1);
    node->right = NULL;
    
    vector_destroy(node->vec);
    free(node->vec);
    node->vec = NULL;
    free(node);
    node = NULL;
};




#endif // __DT_H__