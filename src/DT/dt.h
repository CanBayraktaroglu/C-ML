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

typedef struct DT_Classifier{
    DT_Node* root;
    unsigned char num_classes;
    Dataset* total_dataset;
    Dataset* train_dataset;
    Dataset* test_dataset;
}DT_Classifier;

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

DT_Classifier* dt_classifier_create(){
    DT_Classifier* classifier = (DT_Classifier*)malloc(sizeof(DT_Classifier));
    classifier->root = NULL;
    classifier->num_classes = 0;
    classifier->total_dataset = NULL;
    classifier->train_dataset = NULL;
    classifier->test_dataset = NULL;
    return classifier;
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

void dt_classifier_destroy(DT_Classifier** classifier){
    if (*classifier == NULL) return;
    dt_node_destroy(&(*classifier)->root, 0);
    dataset_destroy(&(*classifier)->total_dataset);
    dataset_destroy(&(*classifier)->train_dataset);

    for (unsigned short i = 0; i < (*classifier)->test_dataset->vec->size; i++){
        Point* p = vector_at((*classifier)->test_dataset->vec, i);
        vec_point_destroy(&p);
    }

    dataset_destroy(&(*classifier)->test_dataset);
    free(*classifier);
    *classifier = NULL;
};

void dt_classifier_set_root(DT_Classifier* classifier, DT_Node* root){
    if (classifier->root == NULL){
        classifier->root = (DT_Node*)malloc(sizeof(DT_Node));
    }
    memcpy(classifier->root, root, sizeof(DT_Node));
};

void dt_classifier_set_total_dataset(DT_Classifier* classifier, Dataset* total_dataset){
    if (classifier->total_dataset == NULL){
        classifier->total_dataset = (Dataset*)malloc(sizeof(Dataset));
    }
    memcpy(classifier->total_dataset, total_dataset, sizeof(Dataset));
};

void dt_classifier_set_train_dataset(DT_Classifier* classifier, Dataset* train_dataset){
    if (classifier->train_dataset == NULL){
        classifier->train_dataset = (Dataset*)malloc(sizeof(Dataset));
    }
    memcpy(classifier->train_dataset, train_dataset, sizeof(Dataset));
};

void dt_classifier_set_test_dataset(DT_Classifier* classifier, Dataset* test_dataset){
    if (classifier->test_dataset == NULL){
        classifier->test_dataset = (Dataset*)malloc(sizeof(Dataset));
    }
    memcpy(classifier->test_dataset, test_dataset, sizeof(Dataset));
};

void dt_node_set_vec(DT_Node* node, Vector* vec){
    if (node->vec == NULL){
        node->vec = (Vector*)malloc(sizeof(Vector));
    }
    memcpy(node->vec, vec, sizeof(Vector));
};

void dt_classifier_build(DT_Node* root, Vector* vec, unsigned char num_classes){
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
    dt_classifier_build(root->left, best_left, num_classes);
    // go right
    dt_classifier_build(root->right, best_right, num_classes);

    free(best_left);
    free(best_right);

};

void dt_classifier_print(DT_Node* node){
    if (node == NULL) return;
    if (node->is_leaf){
        printf("Leaf\n");
        return;
    }

    printf("Feature idx: %hhu\n", node->feature_idx);
    printf("Threshold: %f\n", node->threshold);
    printf("Info gain: %f\n", node->info_gain);
    printf("Left:\n");
    dt_classifier_print(node->left);
    printf("Right:\n");
    dt_classifier_print(node->right);
};

unsigned char dt_classifier_predict(DT_Node* node, Point* p){
    if (node == NULL) return 255;
    if (node->is_leaf){
        unsigned char predicted_class = node->vec->data[0].class;
        return predicted_class;
    }

    if (p->point[node->feature_idx] <= node->threshold){
        return dt_classifier_predict(node->left, p);
    }else{
        return dt_classifier_predict(node->right, p);
    }
};

Metrics* dt_classifier_evaluate(DT_Classifier* dt_classifier){
    if (dt_classifier->test_dataset == NULL){
        printf("Test dataset does not exist.\n");
        exit(1);
    }

	unsigned short int num_classes = (unsigned short int)dt_classifier->test_dataset->num_classes;
    unsigned short int N = dt_classifier->test_dataset->vec->size;
    unsigned short int* true_positives = calloc(num_classes, sizeof(unsigned short int));
    unsigned short int* false_positives = calloc(num_classes, sizeof(unsigned short int));
    unsigned short int* correct = calloc(num_classes, sizeof(unsigned short int));

    for (unsigned short int i = 0; i < N; i++){
        Point* p = vector_at(dt_classifier->test_dataset->vec, i);
	
        int predicted_class = dt_classifier_predict(dt_classifier->root, p);
        if (predicted_class == p->class){
            correct[predicted_class]++;
            true_positives[predicted_class]++;
        }else{
            false_positives[predicted_class]++;
        }
    }

    Metrics* metrics = create_metrics(num_classes);
    for (unsigned short int j = 0; j < num_classes; j++){
        metrics->accuracy += (float)correct[j];
        metrics->precision[j] = (float)true_positives[j] / (true_positives[j] + false_positives[j]);
    }

	metrics->accuracy /= (float)N;

    free(true_positives);
    free(false_positives);
    free(correct);

    return metrics;
};

void dt_classifier_run(){
    //Instantiate a decision tree
	DT_Node* root = dt_node_create();
	//Instantiate a dataset
	Dataset* total_dataset = dataset_create();
	Dataset* train_dataset = dataset_create();
	Dataset* test_dataset = dataset_create();

	// Configs
	DT_Config config;
	load_yaml_dt("../src/DT/configs/configs.yaml", &config);// load

	//Initialize the dataset
	dataset_read_csv(total_dataset, config.data_path);
	printf("Dataset initialized\n");

	// Create DT Classifier
	DT_Classifier* dt_classifier = dt_classifier_create();
	dt_classifier_set_total_dataset(dt_classifier, total_dataset);
	dt_classifier_set_root(dt_classifier, root);
	dt_classifier->num_classes = total_dataset->num_classes;

	//Split the dataset
	dataset_split(total_dataset, train_dataset, test_dataset, config.split_ratio);

	// Set train and test datasets
	dt_classifier_set_train_dataset(dt_classifier, train_dataset);
	dt_classifier_set_test_dataset(dt_classifier, test_dataset);

	//build decision tree
	dt_classifier_build(dt_classifier->root, dt_classifier->train_dataset->vec, dt_classifier->num_classes);
	printf("Decision tree built\n");

	//Print decision tree
	dt_classifier_print(dt_classifier->root);

    //Evaluate the decision tree
    Metrics* metrics = dt_classifier_evaluate(dt_classifier);
    print_metrics(metrics);

	//Free memory
	dt_classifier_destroy(&dt_classifier);
	free(root);
	free(total_dataset);
	free(train_dataset);
	free(test_dataset);
    metrics_destroy(&metrics);
};




#endif // __DT_H__