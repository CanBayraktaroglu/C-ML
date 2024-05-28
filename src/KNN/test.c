#include "KNN.h"
#include <stddef.h>
#include <stdlib.h>
#include <stdio.h>

int main(){
	// Instantiate a dataset
	Dataset* dataset = dataset_create();

	//Instantiate a KNN 
	KNN* knn = KNN_create();

	unsigned char choice;
	printf("Dataset Creation: 0 for Random, 1 for read from file.\n");
	scanf("%hhu", &choice);

	switch(choice){
		case 0:
			// Populate dataset randomly
			dataset_populate(dataset);
			break;
		case 1:
			// Read dataset from file
			unsigned short int N;
			printf("Enter the number of Points:\n");
			scanf("%hu", &N);
			dataset_read_csv(dataset, "../data/iris.csv",N);
			break;
		default:
			printf("Invalid choice\n");
			break;
	}

	// Set the dataset
	KNN_set_dataset(knn, dataset);

	// Split the dataset into train and test
	Dataset* train_dataset = dataset_create();
	Dataset* test_dataset = dataset_create();

	dataset_split(dataset, train_dataset, test_dataset, 0.8);

	KNN_set_datasets(knn, train_dataset, test_dataset);

	unsigned char k;
	unsigned char c;

	printf("Give k: \n");
	scanf("%hhu", &k);
	KNN_set_K(knn, k);

	// fit on the dataset
	KNN_fit(knn);

	if (choice == 0){
		printf("Give number of classes c: \n");
		scanf("%hhu", &c);
		// Assign random classes uniformly
		//float sigma = (float)(c - 1) / 2;
		//float mu = (float)(c - 1) / 2;
		k_d_tree_assign_classes_uniform(knn->root_node, c);
	}

	k_d_tree_print(knn->root_node, 0);

	printf("---------------------------------------\n");

	float x, y;

	printf("Give x of the new point: \n");
	scanf("%f", &x);
	
	printf("Give y of the new point: \n");
	scanf("%f", &y);


	Point* p = point_create();
	point_set_point(p, &x, &y);


	int predicted_class = KNN_predict(knn, p);
	printf("Predicted class: %i\n", predicted_class);
	
	//Free memory
	KNN_destroy(knn);
	
	point_destroy(&p);

	return 0;

}	
