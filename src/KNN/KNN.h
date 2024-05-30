#ifndef KNN_H_
# define KNN_H_

#include "dataset.h"
#include "k_d_tree.h"
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

typedef struct{
	Dataset* total_dataset;
	Dataset* train_dataset;
	Dataset* test_dataset;
	KDTreeNode* root_node;
	unsigned char K;
}KNN;

KNN* KNN_create(){
	KNN* knn = (KNN*) malloc(sizeof(KNN));
	knn->total_dataset = NULL;
	knn->train_dataset = NULL;
	knn->test_dataset = NULL;
	knn->root_node = NULL;
	return knn;
};

void KNN_set_dataset(KNN* knn, Dataset* total_dataset){
	knn->total_dataset = total_dataset;
};

void KNN_set_datasets(KNN* knn, Dataset* train_dataset, Dataset* test_dataset){
	knn->train_dataset = train_dataset;
	knn->test_dataset = test_dataset;
};

void KNN_set_K(KNN* knn, unsigned char K){
	knn->K = K;
};

void KNN_set_node(KNN* knn, KDTreeNode* root_node){
	//memcpy(knn->root_node, root_node, sizeof(KDTreeNode));
	knn->root_node = root_node;
};


void KNN_fit(KNN* knn){
	if (knn->root_node != NULL || knn->train_dataset == NULL){
		printf("Root node or dataset already exists.\n");
		return;
	}

	knn->root_node = k_d_tree_build(knn->train_dataset->vec, 0);
	
};

int KNN_predict(KNN* knn, Point* p){
	if (knn->root_node == NULL){
		printf("Root node does not exist.\n");
		exit(1);
	}

	MaxHeap heap;
	heap.nodes = (HeapNode*) malloc(((unsigned long)knn->K) * sizeof(HeapNode));
	heap.size = 0;
	heap.capacity = (unsigned short int)(knn->K);

	// Get the K nearest neighbors
	k_d_tree_get_nns(knn->root_node, p, (unsigned short int)knn->K, &heap, 0);
	
	int majority_class = 0;
	int max_count = 0;
	int* count = NULL;

	// Create a hash table to store the classes
	HashTable* ht = create_table(int_hash, int_cmp);
	int* classes = (int*) malloc((unsigned long)(knn->K) * sizeof(int));

	// Count the classes
	for (unsigned short int i = 0; i < heap.size; i++){
		// Get the class of the node
		classes[i] = (int)heap.nodes[i].node->p->class;
		ht_increment(ht, (void*)(classes + i));
	}

	for(unsigned short int j = 0; j < heap.size; j++){
		count = (int*)(ht_search(ht, (void*)(classes + j)));

		if (*count > max_count){
			max_count = *count;
			majority_class = classes[j];
		}
	}	
	
	// Free the memory
	free(heap.nodes);
	heap.nodes = NULL;
	free(classes);
	classes = NULL;
	ht_destroy(ht);
	ht = NULL;

	return majority_class;
};

void KNN_destroy(KNN* knn){
	
	
	dataset_destroy(knn->total_dataset);
	dataset_destroy(knn->train_dataset);

	for (unsigned short int i = 0; i < knn->test_dataset->vec->size; i++){
		Point* p = vector_at(knn->test_dataset->vec, i);
		free(p->point);
		p->point = NULL;
	}

	dataset_destroy(knn->test_dataset);
	k_d_tree_node_destroy(knn->root_node);

	free(knn);
	knn = NULL;
};

Metrics KNN_evaluate(KNN* knn){
    if (knn->test_dataset == NULL){
        printf("Test dataset does not exist.\n");
        exit(1);
    }

	unsigned short int num_classes = (unsigned short int)knn->test_dataset->num_classes;
    unsigned short int N = knn->test_dataset->vec->size;
    unsigned short int* true_positives = calloc(num_classes, sizeof(unsigned short int));
    unsigned short int* false_positives = calloc(num_classes, sizeof(unsigned short int));
    unsigned short int* correct = calloc(num_classes, sizeof(unsigned short int));

    for (unsigned short int i = 0; i < N; i++){
        Point* p = vector_at(knn->test_dataset->vec, i);
		/* printf("p->class: %hu\n", p->class);
		printf("p->point[0]: %f\n", p->point[0]);
		printf("p->point[1]: %f\n", p->point[1]);
		printf("p->point[2]: %f\n", p->point[2]);
		printf("p->point[3]: %f\n", p->point[3]); */
        int predicted_class = KNN_predict(knn, p);
		//printf("Predicted class: %d\n", predicted_class);
        if (predicted_class == p->class){
            correct[predicted_class]++;
            true_positives[predicted_class]++;
        }else{
            false_positives[predicted_class]++;
        }
    }

    Metrics metrics;
	metrics.num_classes = num_classes;
    metrics.accuracy = malloc(num_classes * sizeof(float));
    metrics.precision = malloc(num_classes * sizeof(float));
    for (unsigned short int j = 0; j < num_classes; j++){
        metrics.accuracy[j] = (float)correct[j] / (float)N;
        metrics.precision[j] = (float)true_positives[j] / (true_positives[j] + false_positives[j]);
    }

    free(true_positives);
    free(false_positives);
    free(correct);

    return metrics;
};

#endif
