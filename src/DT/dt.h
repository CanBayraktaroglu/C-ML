#ifndef __DT_H__
#define __DT_H__

#include "stdlib.h"
#include "stdio.h"
#include "string.h"
#include "vector.h"
#include "dataset.h"
#include "utils.h"

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
    node->threshold = __FLT_MAX__;
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
    memcpy(node->vec, vec, sizeof(Vector));
};


void dt_node_destroy(DT_Node** node, size_t depth){
    if (*node == NULL) return;
    
    // Destroy the points
    if (!depth){
        for (unsigned short i = 0; i < (*node)->vec->size; i++){
            Point* p = vector_at((*node)->vec, i);
            vec_point_destroy(&p);
            
        }
    }

    // Destroy left node
    dt_node_destroy(&(*node)->left, depth + 1);
    (*node)->left = NULL;

    // Destroy right node
    dt_node_destroy(&(*node)->right, depth + 1);
    (*node)->right = NULL;
    
    // Destroy the vector
    if (depth) vector_destroy(&(*node)->vec);
    else free((*node)->vec);
    (*node)->vec = NULL;

    // Destroy the node itself
    free(*node);
    *node = NULL;
};

void dt_build(DT_Node* root, Vector* vec, unsigned char num_classes){
    if (root == NULL) return;
    if (root->is_leaf) return;
    if (!vec->size) return;

    // Set the vector
    dt_node_set_vec(root, vec);

    // Get number of classes in the vector
    unsigned short num_classes_vec = calculate_num_classes(vec, num_classes);

    //Check if the node is a leaf
    if (num_classes_vec == 1){
        root->is_leaf = 1;
        return;
    }
    
    float best_info_gain = 0.0f;
    unsigned short best_feature_idx = 0;
    float best_threshold = 0;
    Vector* best_left = NULL;
    Vector* best_right = NULL;
    unsigned char dim = vector_at(vec, 0)->dim;

    // for each feature dimension
    for (unsigned char i = 0; i < dim; i++){

        // for each point in the dataset
        for (unsigned short j = 0; j < vec->size; j++){
            Point* p = vector_at(vec, j);
            float threshold = p->point[i];
            Vector* left = vector_create(vec->size / 2);
            Vector* right = vector_create(vec->size / 2);
            
            // partition the dataset into two parts according to the threshold
            for (unsigned short k = 0; k < vec->size; k++){
                Point* q = vector_at(vec, k);
                if (q->point[i] <= threshold){
                    vector_push_back(left, q);
                }else{
                    vector_push_back(right, q);
                }
            }
            float info_gain = calculate_info_gain(vec, left, right, num_classes);

            if (info_gain > best_info_gain){
                best_info_gain = info_gain;
                best_feature_idx = i;
                best_threshold = threshold;
                if (best_left != NULL) vector_destroy(&best_left);
                best_left = left;
                if (best_right != NULL) vector_destroy(&best_right);
                best_right = right;
            }else{
                vector_destroy(&left);
                vector_destroy(&right);
            }

        } 
    }

    root->feature_idx = best_feature_idx;
    root->threshold = best_threshold;
    root->info_gain = best_info_gain;
    root->left = dt_node_create();
    root->right = dt_node_create();

    // go left
    dt_build(root->left, best_left, num_classes);
    // go right
    dt_build(root->right, best_right, num_classes);

    free(best_left);
    free(best_right);

};

void dt_print(DT_Node* node){
    if (node == NULL) return;
    if (node->is_leaf){
        printf("Leaf\n");
        return;
    }

    printf("Feature idx: %hhu\n", node->feature_idx);
    printf("Threshold: %f\n", node->threshold);
    printf("Info gain: %f\n", node->info_gain);
    printf("Left:\n");
    dt_print(node->left);
    printf("Right:\n");
    dt_print(node->right);
};


#endif // __DT_H__