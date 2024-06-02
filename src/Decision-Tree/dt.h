#ifndef __DT_H__
#define __DT_H__

#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "vector.h"
#include "dataset.h"

typedef struct DT_Node{
    u_int8_t feature_idx;
    float threshold;
    struct DT_Node* left;
    struct DT_Node* right;
    float info_gain;
    u_int8_t is_leaf;
    Vector* vec;
}DT_Node;

DT_Node* dt_node_create(){
    DT_Node* node = (DT_Node*)malloc(sizeof(DT_Node));
    node->feature_idx = __UINT8_MAX__;
    node->threshold = MAXFLOAT;
    node->left = NULL;
    node->right = NULL;
    node->vec = NULL;
    node->is_leaf = 0;
    node->info_gain = 0.0f;
    
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

void dt_node_build(DT_Node* root, Vector* vec, unsigned short num_classes){
    if (root == NULL) return;
    if (root->is_leaf) return;
    if (!vec->size) return;

    dt_node_set_vec(root, vec);

    //Check if the node is a leaf
    if (num_classes == 1){
        root->is_leaf = 1;
        return;
    }
    
    float best_info_gain = 0;
    size_t best_feature_idx = 0;
    float best_threshold = 0;
    Vector* best_left = NULL;
    Vector* best_right = NULL;
    unsigned char dim = vector_at(vec, 0)->dim;

    class_freq = calculate_class_frequency(dataset);

    for (unsigned char i = 0; i < dim; i++){
        for (unsigned short j = 0; j < dataset->vec->size; j++){
            Point* p = vector_at(dataset->vec, j);
            float threshold = p->point[i];
            Vector* left = vector_create(0);
            Vector* right = vector_create(0);
            
            for (unsigned short k = 0; k < dataset->vec->size; k++){
                Point* q = vector_at(dataset->vec, k);
                if (q->point[i] <= threshold){
                    vector_push_back(left, q);
                }else{
                    vector_push_back(right, q);
                }
            }
            
            float info_gain = dataset_info_gain(dataset, left, right);
            if (info_gain > best_info_gain){
                best_info_gain = info_gain;
                best_feature_idx = i;
                best_threshold = threshold;
                best_left = left;
                best_right = right;
            }else{
                vector_destroy(left);
                left = NULL;
                vector_destroy(right);
                right = NULL;
            }
        }
    }



    
};


#endif // __DT_H__