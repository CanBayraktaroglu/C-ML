#ifndef K_D_TREE_H_
#define K_D_TREE_H_

#include "vector.h"
#include "utils.h"
#include <stdarg.h>
#include <stdbool.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

// Define the K-D tree node structure
typedef struct KDTreeNode {
    Point* p;
    struct KDTreeNode* left;
    struct KDTreeNode* right;
    unsigned char class;
} KDTreeNode;

KDTreeNode* k_d_tree_node_create(Point* point){
    KDTreeNode* node = (KDTreeNode*) malloc(sizeof(KDTreeNode));
    node->p = (Point*) malloc(sizeof(Point));
    memcpy(node->p, point, sizeof(Point));
    node->left = NULL;
    node->right = NULL;
    node->class = point->class;

    return node;
};

void k_d_tree_node_destroy(KDTreeNode* node){
    if (node == NULL) return;
    //printf("Destroying node\n");
    k_d_tree_node_destroy(node->left);
    k_d_tree_node_destroy(node->right);
    
    //point_destroy(&node->p);
    free(node->p);
    node->p = NULL;
    free(node);
    node = NULL;
};

KDTreeNode* k_d_tree_build(Vector* points, unsigned short int depth){
    if (points->size == 0) return NULL;

    unsigned char axis = (unsigned char)(depth % 2); //alternate between axis

    if (!axis) qsort_x(points, 0, points->size - 1);
    else qsort_y(points, 0, points->size - 1);

    //Find the median point along the current axis#
    unsigned short median_idx = points->size / 2;
    Point* p = vector_at(points, median_idx);

    //Create new k-d tree
    KDTreeNode* node = k_d_tree_node_create(p);

    //Partition the points around the median point
    Vector* left_points = vector_create(points->size >> 1);
    Vector* right_points = vector_create(points->size >> 1);

    for (unsigned short i = 0; i < points->size; i++){
        if (i < median_idx) vector_push_back(left_points, vector_at(points, i));
        else if (i > median_idx) vector_push_back(right_points, vector_at(points, i));
    }

    //memcpy(node->left, k_d_tree_build(left_points, depth+1), sizeof(KDTreeNode));
    node->left = k_d_tree_build(left_points, depth+1);
    //memcpy(node->right, k_d_tree_build(right_points, depth+1), sizeof(KDTreeNode));
    node->right = k_d_tree_build(right_points, depth+1);
    
    //printf("Destroying left vector at depth: %hhu\n", depth);
    //printf("vector size: %hu\n", left_points->size);
    vector_destroy(left_points);
    left_points = NULL;

    //printf("Destroying right vector at depth: %hhu\n", depth);
    //printf("vector size: %hu\n", right_points->size);
    vector_destroy(right_points);
    right_points = NULL;

    return node;

};

void k_d_tree_print(KDTreeNode* node, unsigned short int depth){
    if (node == NULL) return;

    for (int i = 0; i < depth; i++) {
        printf("   ");
    }

    printf("(%.3f, %.3f, %hhu)\n", point_get_x(node->p), point_get_y(node->p), node->class);

    k_d_tree_print(node->left, depth + 1);
    k_d_tree_print(node->right, depth + 1);
};

void k_d_tree_insert(KDTreeNode** root, Point* p, unsigned char depth){
    if (*root == NULL){
        *root = k_d_tree_node_create(p);
        return;
    }

    unsigned char axis = depth % 2;

    if (p->point[axis] < (*root)->p->point[axis]){
        if ((*root)->left == NULL) (*root)->left = k_d_tree_node_create(p);
        else k_d_tree_insert(&((*root)->left), p, depth + 1);
    }
    else {
        if ((*root)->right == NULL) (*root)->right = k_d_tree_node_create(p);
        else k_d_tree_insert(&((*root)->right), p, depth + 1);
    }
};

KDTreeNode* k_d_tree_get_closest(Point* target, KDTreeNode* temp, KDTreeNode* root){
    if (temp == NULL) return root;
    if (root == NULL) return temp;

    float dist_temp = point_calc_dist(temp->p, target);
    float dist_root = point_calc_dist(root->p, target);

    if (dist_temp < dist_root) return temp;
    
    return root;
};

bool k_d_tree_check_nns(KDTreeNode* target, KDTreeNode** nns, unsigned char k){
    for (int i = 0; i < k; i++){
        
        if ( *(nns+i) != target) continue;
        
        //printf("(%p, %p)\n", *(nns+i), target); 
        return true;
    }

    return false;
};

KDTreeNode* k_d_tree_get_nn(KDTreeNode* root, Point* p, KDTreeNode** nns, unsigned char depth, unsigned char k){
    if (root == NULL) return root;

    unsigned char axis = depth % 2;
    KDTreeNode* next_branch;
    KDTreeNode* other_branch;

    if (p->point[axis] < root->p->point[axis]){
        next_branch = root->left;
        other_branch = root->right;        
    }else{
        next_branch = root->right;
        other_branch = root->left;
    }

    KDTreeNode* temp = k_d_tree_get_nn(next_branch, p, nns, depth+1, k);
    KDTreeNode* best = k_d_tree_get_closest(p, temp, root);

    float radius = point_calc_dist(p, best->p);
    radius *= radius;
    float dist = p->point[axis] - root->p->point[axis];

    if (radius >= dist * dist){
        temp = k_d_tree_get_nn(other_branch, p, nns, depth+1, k);
        best = k_d_tree_get_closest(p, temp, best);
    }

    return best;
};

void k_d_tree_search(KDTreeNode* root, Point* target, KDTreeNode** nns, unsigned char* num_nns, unsigned char depth, unsigned char k){
    if (root == NULL) return;

    unsigned char axis = depth % 2;

    // Calc l2 dist between point and target
    float dist = point_calc_dist(root->p, target);

    // if the heap is not full, add the node to the heap
    if (*num_nns < k){
        nns[*num_nns] = root;
        (*num_nns)++;
    }
    else {
        // if the heap is full, check if the current node is closer than the farthest node in the heap
        KDTreeNode* farthest = nns[k - 1];
        float farthest_dist = point_calc_dist(farthest->p, target);
        if (dist < farthest_dist){
            nns[k - 1] = root;
        }
    }

    // Determine which subtree first
    if (target->point[axis] < root->p->point[axis]){
        k_d_tree_search(root->left, target, nns, num_nns, depth + 1, k);
    }
    else {
        k_d_tree_search(root->right, target, nns, num_nns, depth + 1, k);
    }
    
};

void k_d_tree_get_nns(KDTreeNode* root, Point* target, unsigned short int k, MaxHeap* heap, unsigned char depth){
    if (root == NULL) return;

    float dist = point_calc_dist(root->p, target);
    HeapNode hn = { dist, root};
    insert_max_heap(heap, &hn);

    unsigned char axis = depth % 2;
    KDTreeNode* nextBranch = NULL;
    KDTreeNode* otherBranch = NULL;

    if (target->point[axis] < root->p->point[axis]) {
        nextBranch = root->left;
        otherBranch = root->right;
    } else {
        nextBranch = root->right;
        otherBranch = root->left;
    }

    k_d_tree_get_nns(nextBranch, target, k, heap, depth + 1);

    if (heap->size < k || fabsf(root->p->point[axis] - target->point[axis]) < heap->nodes[0].dist) {
        k_d_tree_get_nns(otherBranch, target, k, heap, depth + 1);
    }
};

void k_d_tree_assign_classes_gaussian(KDTreeNode* root, float* mu, float* sigma){
    if (root == NULL) return;

    unsigned char class = normalRandom() * *sigma + *mu;
    root->class = class;
    root->p->class = class;
    
    k_d_tree_assign_classes_gaussian(root->left, mu, sigma);
    k_d_tree_assign_classes_gaussian(root->right, mu, sigma);
};

void k_d_tree_assign_classes_uniform(KDTreeNode* root, unsigned char c){
    if (root == NULL) return;

    unsigned char class = randomGenerator() * (float)c;
    root->class = class;
    root->p->class = class;

    k_d_tree_assign_classes_uniform(root->left, c);
    k_d_tree_assign_classes_uniform(root->right, c);
};

#endif