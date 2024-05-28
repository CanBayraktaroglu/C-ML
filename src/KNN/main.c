#include "KNN.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

int main(){
	// Instantiate a dataset
	Dataset* dataset = dataset_create();

	//Instantiate a KNN 
	KNN* knn = KNN_create();

	unsigned short int N;
	printf("Enter the number of Points:\n");
	scanf("%hu", &N);
	dataset_read_csv(dataset, "../src/KNN/data/iris.csv",N);
		
	// Set the dataset
	KNN_set_dataset(knn, dataset);

	// Split the dataset into train and test
	Dataset* train_dataset = dataset_create();
	Dataset* test_dataset = dataset_create();

	float split_ratio;
	printf("Set Split Ratio: \n");
	scanf("%f", &split_ratio);

	dataset_split(dataset, train_dataset, test_dataset, split_ratio);

	KNN_set_datasets(knn, train_dataset, test_dataset);

	unsigned char k;
	unsigned char c;

	printf("Give k: \n");
	scanf("%hhu", &k);
	KNN_set_K(knn, k);

	// fit on the dataset
	KNN_fit(knn);

	k_d_tree_print(knn->root_node, 0);

	printf("---------------------------------------\n");

	// Evaluate the model
	Metrics metrics = KNN_evaluate(knn);
	print_metrics(&metrics);
	
	//Free memory
	KNN_destroy(knn);
	metrics_destroy(&metrics);
	
	return 0;

}	
